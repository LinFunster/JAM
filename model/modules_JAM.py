import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def inverse_rotate_by_yaw(traj,yaw):
    """
    traj: [..., 2]
    yaw: [B, 1]
    return:
        traj_rot: [..., 2] 
    """
    """
    traj torch.Size([B, 11, 2])
    yaw torch.Size([B, 1])
    rot_matrix torch.Size([B, 2, 2])
    """
    # rotate xy
    cosa = torch.cos(-yaw) # [B, 1] 
    sina = torch.sin(-yaw) # [B, 1]
    shape_list = [1 for i in range(len(traj.shape)-3)] 
    if len(shape_list) == 0:
        rot_matrix = torch.stack((
            cosa, sina,
            -sina, cosa
        ), dim=1).view(-1, 2, 2).float() # [B, 2, 2] 
    else:
        rot_matrix = torch.stack((
            cosa, sina,
            -sina, cosa
        ), dim=1).view(-1,*shape_list, 2, 2).float() # [B, 2, 2] 
    traj_rot = torch.matmul(traj, rot_matrix) # [..., 2]@[B, 2, 2]  -> [..., 2] 
    # rotate v_xy

    return traj_rot

def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/sshaoshuai/MTR
    """
    # input pos_tensor: torch.Size([32, 11, 6, 4, 2])
    half_hidden_dim = hidden_dim // pos_tensor.size(-1)
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale # torch.Size([32, 11, 6, 4])
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t # torch.Size([32, 11, 6, 4, 128])
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2) # > torch.Size([32, 11, 6, 4, 64, 2]) > torch.Size([32, 11, 6, 4, 128])
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=-1) # torch.Size([32, 11, 6, 4, 256])
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[..., 2] * scale
        pos_w = w_embed[..., None] / dim_t
        pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1).flatten(-2)

        h_embed = pos_tensor[..., 3] * scale
        pos_h = h_embed[..., None] / dim_t
        pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(-2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

def rotate_refinement_traj_by_yaw(traj,yaw):
    """
    traj: [..., 2]
    yaw: [B, 1]
    return:
        traj_rot: [..., 2] 
    """
    """
    traj torch.Size([B, M, 80, 2])
    yaw torch.Size([B, 1])
    rot_matrix torch.Size([B, 1, 2, 2])
    """
    cosa = torch.cos(yaw) # [B, 1] 
    sina = torch.sin(yaw) # [B, 1] 
    rot_matrix = torch.stack((
        cosa, sina,
        -sina, cosa
    ), dim=1).view(-1, 1, 2, 2).float() # [B, 2, 2] 
    traj_rot = torch.matmul(traj, rot_matrix) # [..., 2]@[B, 2, 2]  -> [..., 2] 
    return traj_rot

class PositionalEncoding(nn.Module):
    def __init__(self, max_len=100):
        super(PositionalEncoding, self).__init__()
        d_model = 256
        dropout = 0.1
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe
        
        return self.dropout(x)
    
class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.pre_mlp = nn.LSTM(8, 256, 2, batch_first=True)
        self.type_emb = nn.Embedding(4, 256, padding_idx=0)

    def inverse_rotate_input_agent_traj_by_yaw(self,traj):
        """
        traj: [B, 11, 9]
        center_x, center_y, heading, 
        velocity_x,  velocity_y,  length, 
        width, height, object_type
        yaw: [B, 1]
        return:
            traj_rot: [B, 11, 9] 
        """
        """
        traj torch.Size([B, 11, 9])
        yaw torch.Size([B, 1])
        rot_matrix torch.Size([B, 2, 2])
        """
        # cur pos
        current_pos = traj[...,-1,:2].clone() # [B, 2]
        heading = traj[...,-1, 2:3].clone() # [B, 1]
        rot_traj = traj.clone() # [B, 11, 9]
        origin_pos = torch.concat([current_pos, heading],dim=-1).unsqueeze(-2) # [B, 1, 3]
        current_pos = torch.concat([current_pos, torch.cos(heading), torch.sin(heading)],dim=-1) # [B, 4]
        # norm yaw&pos
        rot_traj[...,:3] -= origin_pos
        # rotate xy
        rot_traj[...,:2] = inverse_rotate_by_yaw(traj[...,:2],heading)
        # rotate v_xy
        rot_traj[...,3:5] = inverse_rotate_by_yaw(traj[...,3:5],heading)
        return rot_traj, current_pos

    def forward(self, input_traj): # [B,T,9]
        # rotate traj
        rot_traj, current_pos = self.inverse_rotate_input_agent_traj_by_yaw(input_traj)
        cur_pos = gen_sineembed_for_position(current_pos)[:,None] # [B,1,256] 
        # traj_feature
        traj_feature = self.pre_mlp(rot_traj[:, :, :8])[0][:, -1:] # [B, 1, 256]
        # type_feature
        type_feature = self.type_emb(input_traj[:, -1:, 8].int()) # [B, 1, 256]
        # traj2pos
        traj2type2pos = traj_feature+type_feature+cur_pos
        return traj2type2pos[:,0], cur_pos

class AgentEncoder_GF(nn.Module):
    def __init__(self):
        super(AgentEncoder_GF, self).__init__()
        self.motion = nn.LSTM(8, 256, 2, batch_first=True)
        self.type_emb = nn.Embedding(4, 256, padding_idx=0)

    def forward(self, inputs):
        traj, _ = self.motion(inputs[:, :, :8])
        output = traj[:, -1]
        type = self.type_emb(inputs[:, -1, 8].int())
        output = output + type

        return output

class LaneEncoder(nn.Module):
    def __init__(self):
        super(LaneEncoder, self).__init__()
        # encdoer layer
        self.self_line = nn.Linear(3, 128)
        self.left_line = nn.Linear(3, 128)
        self.right_line = nn.Linear(3, 128)
        self.speed_limit = nn.Linear(1, 64)
        self.self_type = nn.Embedding(4, 64, padding_idx=0)
        self.left_type = nn.Embedding(11, 64, padding_idx=0)
        self.right_type = nn.Embedding(11, 64, padding_idx=0)
        self.traffic_light_type = nn.Embedding(9, 64, padding_idx=0)
        self.interpolating = nn.Embedding(2, 64)
        self.stop_sign = nn.Embedding(2, 64)
        # hidden layers
        self.line2type_mlp = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, 256))

    def inverse_rotate_input_poly_line_by_yaw(self, polyline):
        """
        crosswalk: [B, N, 6, 100, 16]
        self_point (x, y, h), 
        left_boundary_point (x, y, h), 
        right_boundary_point (x, y, h), 
        speed limit (float),
        self_type (int), left_boundary_type (int), right_boundary_type (int),
        traffic light (int), interpolating (bool), stop_sign (bool)
        yaw: [B*N*6, 1]
        return:
            crosswalk_rot: [..., 3] 
        """
        # current state
        B,N,N_line,N_point,F = polyline.shape 
        polyline = polyline.view(B*N*N_line,N_point,F) # [B*N*6, 100, 16]
        self_current_pos = polyline[...,0,:2].clone() # [B*N*6, 2]
        self_heading = polyline[:,0,2:3].clone() # [B*N*6, 1]
        rot_polyline = polyline.clone()
        origin_pos = torch.concat([self_current_pos, self_heading],dim=-1).unsqueeze(-2) # [B*N*6, 1, 3]
        current_pos = torch.concat([self_current_pos, torch.cos(self_heading), torch.sin(self_heading)],dim=-1) # [B*N*6, 4]
        # rotate selfline&left_boundary_point&right_boundary_point xy&yaw
        rot_polyline[...,:3] = polyline[...,:3] - origin_pos
        rot_polyline[...,3:6] = polyline[...,3:6] - origin_pos
        rot_polyline[...,6:9] = polyline[...,6:9] - origin_pos
        rot_polyline[...,:2] = inverse_rotate_by_yaw(polyline[...,:2],self_heading) # [B*N*6, 100] # rotate self xy&yaw
        rot_polyline[...,3:5] = inverse_rotate_by_yaw(polyline[...,3:5],self_heading) # rotate left_boundary_point xy&yaw
        rot_polyline[...,6:8] = inverse_rotate_by_yaw(polyline[...,6:8],self_heading) # rotate right_boundary_point xy&yaw
        # traj_rot
        rot_polyline = rot_polyline.view(B,N,N_line,N_point,F) # [B, N, 6, 100, 16]
        current_pos = current_pos.view(B,N,N_line,1,-1) # [B, N, 6, 1, 4]
        return rot_polyline, current_pos
    
    def forward(self, inputs): # map_lanes torch.Size([1, 11, 6, 100, 16])
        # rotate poly line
        inputs, current_pos = self.inverse_rotate_input_poly_line_by_yaw(inputs)
        # embedding
        self_line = self.self_line(inputs[..., :3])
        left_line = self.left_line(inputs[..., 3:6])
        right_line = self.right_line(inputs[..., 6:9])
        speed_limit = self.speed_limit(inputs[..., 9].unsqueeze(-1))
        self_type = self.self_type(inputs[..., 10].int())
        left_type = self.left_type(inputs[..., 11].int())
        right_type = self.right_type(inputs[..., 12].int())
        traffic_light = self.traffic_light_type(inputs[..., 13].int())
        interpolating = self.interpolating(inputs[..., 14].int())
        stop_sign = self.stop_sign(inputs[..., 15].int())
        # attr fusion
        lane_attr = self_type + left_type + right_type + traffic_light + interpolating + stop_sign
        lane_embedding = torch.cat([self_line, left_line, right_line, speed_limit, lane_attr], dim=-1)
        # line2type
        line2type = self.line2type_mlp(lane_embedding) # torch.Size([B, N, 6, 100, 256])
        # # line2pos
        cur_pos = gen_sineembed_for_position(current_pos) # torch.Size([B, N, 6, 1, 256])
        line2pos = line2type+cur_pos
        return line2pos, cur_pos

class CrosswalkEncoder(nn.Module):
    def __init__(self):
        super(CrosswalkEncoder, self).__init__()
        self.point_net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))

    def inverse_rotate_input_crosswalk_line_by_yaw(self, polyline):
        """
        crosswalk: [B, N, 4, 100, 3]
        center_x, center_y, heading
        yaw: [B*N*4, 1]
        return:
            crosswalk_rot: [..., 3] 
        """
        # current state
        B,N,N_line,N_point,F = polyline.shape 
        polyline = polyline.view(B*N*N_line,N_point,F) # [B*N*4, 100, 3]
        current_pos = polyline[...,0,:2].clone() # [B*N*4, 2]
        heading = polyline[:,0,2:3].clone() # [B*N*4, 1]
        rot_polyline = polyline.clone()
        origin_pos = torch.concat([current_pos, heading],dim=-1).unsqueeze(-2) # [B*N*4, 1, 3]
        current_pos = torch.concat([current_pos, torch.cos(heading), torch.sin(heading)],dim=-1) # [B*N*4, 4]
        # rotate xy
        rot_polyline[...,:3] -= origin_pos # [B*N*4,100,3]
        rot_polyline[...,:2] = inverse_rotate_by_yaw(polyline[...,:2],heading)
        # traj_rot
        rot_polyline = rot_polyline.view(B,N,N_line,N_point,F) # [B, N, 4, 100, 3]
        current_pos = current_pos.view(B,N,N_line,1,-1) # [B, N, 4, 1, 4]
        return rot_polyline, current_pos
    
    def forward(self, input_polyline): # map_crosswalks torch.Size([B, N, 4, 100, 3])
        # rotate polyline
        rot_polyline, current_pos = self.inverse_rotate_input_crosswalk_line_by_yaw(input_polyline)
        cross_embedding = self.point_net(rot_polyline) # torch.Size([B, N, 4, 100, 256])
        # # line2pos
        cur_pos = gen_sineembed_for_position(current_pos) # torch.Size([B, N, 4, 1, 256])
        line2pos = cross_embedding + cur_pos
        return line2pos, cur_pos 
    
class GMMFutureEncoder(nn.Module):
    def __init__(self):
        super(GMMFutureEncoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 256))
        self.type_emb = nn.Embedding(4, 256, padding_idx=0)

    def inverse_rotate_input_agent_traj_by_yaw(self,traj, current_states):
        """
        traj: [B, 2, 6, 80, 10]
        current_states: torch.Size([B, 2, 9])
        center_x, center_y, sigma_x, sigma_y, heading, 
        velocity_x,  velocity_y,  length, 
        width, height, object_type
        yaw: [B, 1]
        return:
            traj_rot: [B, 2, 6, 80, 10] 
        """
        """
        traj torch.Size([B, 2, 80, 9])
        yaw torch.Size([B, 1])
        rot_matrix torch.Size([B, 2, 2])
        """
        # current state
        B,N,M,N_point,F = traj.shape  # torch.Size([64, 2, 6, 80, 10])
        traj = traj.view(B*N*M,N_point,F) # [B*N, 80, 10] torch.Size([768, 80, 10])
        current_states = current_states.clone()[...,None,:].repeat(1,1,M,1).contiguous().view(B*N*M,-1) # [B*N*M, 9]
        current_pos = current_states[...,:2] # [B, 2]
        heading = current_states[...,2:3] # [B, 1]
        rot_traj = traj.clone() # [B*N, 80, 10]
        origin_pos = torch.concat([current_pos, heading],dim=-1).unsqueeze(-2) # [B, 1, 3]
        current_pos = torch.concat([current_pos, torch.cos(heading), torch.sin(heading)],dim=-1) # [B, 4]
        # rotate yaw
        rot_traj[...,:2] -= origin_pos[...,:2]
        rot_traj[...,4] -= origin_pos[...,2]
        # rotate xy
        rot_traj[...,:2] = inverse_rotate_by_yaw(traj[...,:2],heading)
        # rotate v_xy
        rot_traj[...,5:7] = inverse_rotate_by_yaw(traj[...,5:7],heading)
        # traj_rot
        rot_traj = rot_traj.view(B,N,M,N_point,F)
        current_pos = current_pos.view(B,N,M,-1)[:,:,:1,:].contiguous() # [B,N,1,4]
        return rot_traj, current_pos

    def forward(self, trajs, current_states):
        # rotate traj and get pos
        rot_traj, current_pos = self.inverse_rotate_input_agent_traj_by_yaw(trajs,current_states)
        cur_pos = gen_sineembed_for_position(current_pos) # [B,N,1,256] 
        # traj feature 
        traj_feature = self.mlp(rot_traj.detach()) # [B,N,M,T,256] 
        traj_feature = torch.max(traj_feature, dim=-2).values # [B,N,M,256] 
        # type feature 
        type_feature = self.type_emb(current_states[:, :, None, 8].int()) # [B,N,1,256] 
        # traj2type2pos
        traj2type2pos = traj_feature + type_feature + cur_pos
        return traj2type2pos

class KeyPointEncoder(nn.Module):
    def __init__(self):
        super(KeyPointEncoder, self).__init__()
        self.pre_mlp = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
        self.type_emb = nn.Embedding(4, 256, padding_idx=0)

    def forward(self, keypoints, current_states):
        # get pos_feature
        pos_feature = gen_sineembed_for_position(keypoints.detach()) # torch.Size([B, N, M, 4, 256])
        pos_feature = self.pre_mlp(pos_feature) # torch.Size([B, N, M, 4, 256])
        pos_feature = pos_feature.mean(dim=-2) # torch.Size([B, N, M, 256])
        # traj2type
        type_feature = self.type_emb(current_states[:, :, None, 8].int()) # [B,N,1,256] 
        traj2type = pos_feature+type_feature
        return traj2type

class GMMPredictor(nn.Module):
    def __init__(self, future_len):
        super(GMMPredictor, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(nn.Linear(256, 512), nn.ELU(), nn.Dropout(0.1), nn.Linear(512, self._future_len*4))
        self.modal_score = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, input):
        B, M, _ = input.shape
        res = self.gaussian(input).view(B, M, self._future_len, 4) # mu_x, mu_y, log_sig_x, log_sig_y
        modal_score = self.modal_score(input).squeeze(-1) # unzip dim -1
        return res, modal_score


class SelfTransformerKeyPoint(nn.Module):
    def __init__(self):
        super(SelfTransformerKeyPoint, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))
        self.sa_qcontent_proj = nn.Linear(dim, dim)
        self.sa_qpos_proj = nn.Linear(dim, dim)
        self.sa_kcontent_proj = nn.Linear(dim, dim)
        self.sa_kpos_proj = nn.Linear(dim, dim)
        self.sa_v_proj = nn.Linear(dim, dim)

    def forward(self, tgt, keypoint, mask=None):
        # Apply projections here
        q_content = self.sa_qcontent_proj(tgt)
        q_pos = self.sa_qpos_proj(keypoint)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(keypoint)
        value = self.sa_v_proj(tgt)
        query = q_content + q_pos
        key = k_content + k_pos
        # self attention
        attention_output, _ = self.self_attention(query, key, value, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output + value)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output

class SelfTransformer(nn.Module):
    def __init__(self):
        super(SelfTransformer, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, inputs, mask=None):
        attention_output, _ = self.self_attention(inputs, inputs, inputs, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output + inputs)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output

class CrossTransformer(nn.Module):
    def __init__(self):
        super(CrossTransformer, self).__init__()
        heads, dim, dropout = 8, 256, 0.1
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim*4, dim), nn.Dropout(dropout))

    def forward(self, query, key, value, mask=None):
        attention_output, _ = self.cross_attention(query, key, value, key_padding_mask=mask)
        attention_output = self.norm_1(attention_output)
        output = self.norm_2(self.ffn(attention_output) + attention_output)

        return output

class MarginalInitialDecoder(nn.Module):
    def __init__(self, modalities, neighbors, future_len, class_query_N):
        super(MarginalInitialDecoder, self).__init__()
        dim = 256
        self._modalities = modalities*class_query_N
        self.multi_modal_query_embedding = nn.Embedding(self._modalities, dim)
        self.agent_query_embedding = nn.Embedding(neighbors+1, dim)
        self.mode2scene_crs_attn = CrossTransformer()
        self.after_mode2mode_attn = SelfTransformer()
        self.predictor = GMMPredictor(future_len)
        self.register_buffer('modal', torch.arange(self._modalities).long())
        self.register_buffer('agent', torch.arange(neighbors+1).long())

    def forward(self, id, current_state, encoding, mask):
        # encoding torch.Size([32, 114, 256])
        # get query
        multi_modal_query = self.multi_modal_query_embedding(self.modal) # torch.Size([M*Class_N, 256])
        agent_query = self.agent_query_embedding(self.agent[id]) # torch.Size([256])
        multi_modal_agent_query = multi_modal_query + agent_query[None, :] # torch.Size([M*Class_N, 256])
        query = encoding[:, None, id] + multi_modal_agent_query #+ type_query # torch.Size([B, M*Class_N, 256])
        # decode trajectories
        mode2scene = self.mode2scene_crs_attn(query, encoding, encoding, mask) # query_content torch.Size([B, M*Class_N, 256])
        mode2mode =self.after_mode2mode_attn(mode2scene) # [B, M*Class_N, 256]
        predictions, modal_score = self.predictor(mode2mode) # predictions, modal_score torch.Size([B, M*Class_N, 50, 4]) and torch.Size([B, M*Class_N])
        # post process
        predictions[..., :2] = rotate_refinement_traj_by_yaw(predictions[..., :2], current_state[:, 2:3])
        predictions[..., :2] += current_state[:, None, None, :2]

        return mode2mode, predictions, modal_score

class JointInitialDecoder(nn.Module):
    def __init__(self, modalities, neighbors, future_encoder, keypoint_encoder, future_len):
        super(JointInitialDecoder, self).__init__()
        dim = 256
        self.multi_modal_query_embedding = nn.Embedding(modalities, dim)
        self.agent_query_embedding = nn.Embedding(neighbors+1, dim)
        self.register_buffer('modal', torch.arange(modalities).long())
        self.register_buffer('agent', torch.arange(neighbors+1).long())
        self.before_mode2mode_attn = SelfTransformer()
        self.after_mode2mode_attn = SelfTransformer()
        self.agent2agent_attn = SelfTransformer()
        self.mode2scene_crs_attn = CrossTransformer()
        self.future_encoder = future_encoder
        self.keypoint_encoder = keypoint_encoder
        self.decoder = GMMPredictor(future_len)

    def forward(self, id, current_states, actors, scores, keypoints, last_content, encoding, mask):
        # last_content [B, M_topK*Class_N, 256]
        # keypoints [B, N, M_topK*Class_N, K, 2]
        # mask [B, 134]
        # encoding the trajectories from the last level 
        multi_fusion_features = self.future_encoder(actors, current_states) # torch.Size([B, N, M_topK*Class_N, 256])
        multi_keyp_futures = self.keypoint_encoder(keypoints, current_states) # torch.Size([B, N, M_topK*Class_N, 256])
        multi_fusion_features = multi_fusion_features + multi_keyp_futures # [B, N, M_topK*Class_N, 256]
        multi_fusion_features = multi_fusion_features * scores.softmax(-1).unsqueeze(-1)
        # before_mode2mode_attn
        B, N, CQ_N = multi_fusion_features.shape[:3]
        fusion_futures = (multi_fusion_features).view(B,N*CQ_N,-1) # torch.Size([B, N*M_topK*Class_N, 256])
        fusion_mask = mask[:, :N, None].repeat(1,1,CQ_N).contiguous().view(B,N*CQ_N)
        before_mode2mode = self.before_mode2mode_attn(fusion_futures, fusion_mask) # torch.Size([B, N*CQ_N, 256])
        # agent2agent_attn
        before_mode2mode = before_mode2mode.view(B,N,CQ_N,-1).mean(2) # torch.Size([B, N, CQ_N, 256])
        agent2agent = self.agent2agent_attn(before_mode2mode, mask[:, :N]) # torch.Size([B, N, 256])
        # get mode2scene_query
        multi_modal_query = self.multi_modal_query_embedding(self.modal) # torch.Size([M, 256])
        agent_query = self.agent_query_embedding(self.agent[id]) # torch.Size([256])
        multi_modal_agent_query = multi_modal_query[None,:] + agent_query[None, None, :] #+ type_query # torch.Size([B, M, 256])
        last_content = (last_content).mean(dim=1)[:,None] # torch.Size([B, 1, 256])
        query_fusion_futures = (multi_fusion_features[:,id]).mean(dim=1)[:,None] # torch.Size([B, 1, 256])
        mode2scene_query = encoding[:, None, id] + multi_modal_agent_query + query_fusion_futures + last_content # torch.Size([B, M, 256])
        # mode2scene_crs_attn
        mode2scene_kv = torch.cat([agent2agent, encoding], dim=1) # append the interaction encoding to the context encoding
        mode2scene_mask = torch.cat([mask[:, :N], mask], dim=1).clone()
        mode2scene_mask[:, id] = True # mask the agent future itself from last level
        mode2scene = self.mode2scene_crs_attn(mode2scene_query, mode2scene_kv, mode2scene_kv, mode2scene_mask)
        # after_mode2mode
        after_mode2mode = self.after_mode2mode_attn(mode2scene) # torch.Size([B, M, 256])
        # decoding the trajectories
        trajectories, modal_score = self.decoder(after_mode2mode)
        # post process
        trajectories[..., :2] = rotate_refinement_traj_by_yaw(trajectories[..., :2], current_states[:, id, 2:3])
        trajectories[..., :2] += current_states[:, id, None, None, :2]

        return after_mode2mode, trajectories, modal_score
    
    