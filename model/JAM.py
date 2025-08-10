import torch
from .modules_JAM import *

class Encoder(nn.Module):
    def __init__(self, neighbors_to_predict, layers=6):
        super(Encoder, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self._neighbors = neighbors_to_predict
        self.agent_encoder = AgentEncoder()
        self.lane_encoder = LaneEncoder()
        self.crosswalk_encoder = CrosswalkEncoder()
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation=F.gelu, dropout=dropout, batch_first=True)
        self.agent2map_attn = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)
        self.agent2agent_attn = SelfTransformer()

    def segment_map(self, map, map_encoding):
        """Downsampling the map encoding."""
        stride = 10
        B, N_e, N_p, D = map_encoding.shape

        # segment mask   map [B, 4, 100, 16]
        map_mask_raw = torch.eq(map, 0)[:, :, :, 0] # [B,4,100]
        map_mask = map_mask_raw.reshape(B, N_e, N_p//stride, N_p//(N_p//stride)) # [B,4,10,10]
        map_mask = torch.min(map_mask, dim=-1)[0].reshape(B, -1)

        # segment map
        map_valid = map_mask_raw==False
        map_encoding_valid = torch.zeros_like(map_encoding)
        map_encoding_valid[map_valid] = map_encoding[map_valid]
        map_encoding = F.max_pool2d(map_encoding_valid.permute(0, 3, 1, 2), kernel_size=(1, stride))
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)

        return map_encoding, map_mask
    
    def forward(self, inputs):
        """
        ego_state torch.Size([1, 11, 9])
        neighbors_state torch.Size([1, 10, 11, 9])
        map_lanes torch.Size([1, 11, 6, 100, 16])
        map_crosswalks torch.Size([1, 11, 4, 100, 3])
        ref_line torch.Size([1, 1000, 5])
        """
        # agent encoding
        ego = inputs['ego_state']
        neighbors = inputs['neighbors_state']
        agents = torch.cat([ego.unsqueeze(1), neighbors], dim=1) # torch.Size([B, N+N_nei, 11, 9])
        agents_N = agents.shape[1]
        feature_agents = []
        for i in range(agents_N): # idx 0 is ego
            feature_agents_i, cur_pos_agents_i = self.agent_encoder(agents[:,i]) # [B,11,256] [B,1,256]
            feature_agents.append(feature_agents_i)
        feature_agents = torch.stack(feature_agents, dim=1) # torch.Size([B, N+N_nei, 256])
        mask_agents = torch.eq(agents[:, :, -1].sum(-1), 0) # torch.Size([B, N+N_nei])

        # agent2agent
        agent2agent = self.agent2agent_attn(feature_agents,mask=mask_agents)   

        # map encoding
        lanes = inputs['map_lanes']
        crosswalks = inputs['map_crosswalks']
        feature_lanes,cur_pos_lanes = self.lane_encoder(lanes) # torch.Size([B, N, 6, 100, 256]), [B, N, 6, 1, 256]
        feature_crosswalks,cur_pos_crosswalks = self.crosswalk_encoder(crosswalks) # torch.Size([B, N, 4, 100, 256]) , [B, N, 4, 1, 256]
        
        # attention fusion
        features = []
        masks = []
        N = self._neighbors + 1
        assert agents_N >= N, 'Too many neighbors to predict'
        for i in range(N):
            # map lane
            feature_lanes_i, mask_lanes_i = self.segment_map(lanes[:, i], feature_lanes[:, i]) # torch.Size([B, 6*10, 256])
            # map crosswalk
            feature_crosswalks_i, mask_crosswalks_i = self.segment_map(crosswalks[:, i], feature_crosswalks[:, i]) # torch.Size([B, 4*10, 256])
            # agent2map
            fusion_input = torch.cat([agent2agent, feature_lanes_i, feature_crosswalks_i], dim=1) # torch.Size([B, 100+N+N_nei, 256])
            fusion_mask = torch.cat([mask_agents, mask_lanes_i, mask_crosswalks_i], dim=1) # torch.Size([B, 100+N+N_nei])
            agent2map = self.agent2map_attn(fusion_input, src_key_padding_mask=fusion_mask)
            # store info
            masks.append(fusion_mask)
            features.append(agent2map)
        
        # outputs
        encodings = torch.stack(features, dim=1)
        masks = torch.stack(masks, dim=1)
        encoder_outputs = {
            'agents': agents, # torch.Size([B, N, T, 9])
            'encodings': encodings, # torch.Size([B, N, 100+N+N_nei, 256])
            'masks': masks # torch.Size([B, N, 100+N+N_nei])
        }

        return encoder_outputs

class Decoder(nn.Module):
    def __init__(self, modalities,class_query_topK,future_len, neighbors_to_predict, class_query_N):
        super(Decoder, self).__init__()
        self._neighbors = neighbors_to_predict
        future_encoder = GMMFutureEncoder()
        keypoint_encoder = KeyPointEncoder()
        self.pre_initial_stage = MarginalInitialDecoder(class_query_topK, neighbors_to_predict, future_len, class_query_N)
        self.initial_stage = JointInitialDecoder(modalities, neighbors_to_predict, future_encoder, keypoint_encoder,future_len)

    def find_keypoint(self, last_level_traj): # torch.Size([32, 11, 6, 50, 10]) torch.Size([32, 11, 6])
        """
        last_level_traj (10dim): mu_x, mu_y, sigma_x, sigma_y, heading, velocity_x,  velocity_y, length, width, height
        # find keypoint
        # rule1: find the collision point&interaction point: closest point of the crossing trajectories
        # rule2: find the destination intent point
        """
        # return trajs
        last_level_traj = torch.concat([last_level_traj[...,:2],last_level_traj[...,5:7]],dim=-1) # .detach()
        B,N,M,T,P = last_level_traj.shape
        # get all trajs
        all_trajs = last_level_traj[:, torch.arange(N)[None, :].repeat(N, 1), :, :, :] # torch.Size([32, 11, 11, 6, 50, 2])
        """destination keypoint"""
        # keypoints torch.Size([32, 11, 11, 6, 4, 2])
        keypoints = torch.concat([all_trajs[...,30:31,:], all_trajs[...,50:51,:], all_trajs[...,-1:,:]], dim=-2)
        return keypoints
     
    def gmm_state_process(self, trajs_gmm, current_states):
        trajs = trajs_gmm[...,:2]
        M = trajs.shape[2]
        current_states = current_states.unsqueeze(2).expand(-1, -1, M, -1)
        xy = torch.cat([current_states[:, :, :, None, :2], trajs], dim=-2)
        dxy = torch.diff(xy, dim=-2)
        v = dxy / 0.1
        theta = torch.atan2(dxy[..., 1], dxy[..., 0].clamp(min=1e-3)).unsqueeze(-1)
        T = trajs.shape[3]
        size = current_states[:, :, :, None, 5:8].expand(-1, -1, -1, T, -1)
        trajs = torch.cat([trajs_gmm, theta, v, size], dim=-1) # (mu_x, mu_y, sigma_x, sigma_y, heading, vx, vy, w, l, h)
        return trajs
    
    def forward(self, encoder_inputs):
        # input data
        decoder_outputs = {}
        N = self._neighbors + 1
        assert encoder_inputs['agents'].shape[1] >= N, 'Too many neighbors to predict'
        current_states = encoder_inputs['agents'][:, :, -1]
        encodings, masks = encoder_inputs['encodings'], encoder_inputs['masks']
        # marginal initial
        results = [self.pre_initial_stage(i, current_states[:, i], encodings[:, i], masks[:, i]) for i in range(N)]
        last_content = torch.stack([result[0] for result in results], dim=1)
        last_level_gmm = torch.stack([result[1] for result in results], dim=1)
        last_modal_scores = torch.stack([result[2] for result in results], dim=1)
        decoder_outputs['marginal_interactions'] = last_level_gmm
        decoder_outputs['marginal_scores'] = last_modal_scores
        # joint refinement
        last_level_traj = self.gmm_state_process(last_level_gmm, current_states[:, :N])
        keypoints = self.find_keypoint(last_level_traj) # keypoints torch.Size([B, N, N, M, 4, 2])
        results = [self.initial_stage(i, current_states[:, :N], last_level_traj, last_modal_scores, keypoints[:, i], \
                    last_content[:, i], encodings[:, i], masks[:, i]) for i in range(N)]
        last_content = torch.stack([result[0] for result in results], dim=1)
        last_level_gmm = torch.stack([result[1] for result in results], dim=1)
        last_modal_scores = torch.stack([result[2] for result in results], dim=1)
        decoder_outputs['joint_interactions'] = last_level_gmm
        decoder_outputs['joint_scores'] = last_modal_scores
        return decoder_outputs


class JAM(nn.Module):
    def __init__(self, modalities,class_query_topK,neighbors_to_predict, future_len, encoder_layers=6, class_query_N=8):
        super(JAM, self).__init__()
        self.encoder = Encoder(neighbors_to_predict, encoder_layers)
        self.decoder = Decoder(modalities,class_query_topK,future_len, neighbors_to_predict, class_query_N)

    def forward(self, inputs): 
        encoder_outputs = self.encoder(inputs)
        outputs = self.decoder(encoder_outputs)
        
        return outputs
