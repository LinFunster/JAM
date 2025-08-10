import sys
sys.path.append("..")
import glob
import torch
import argparse
import logging
import pandas as pd
import numpy as np
from data_process import *
from utils.visualization_utils import *
from utils.inter_pred_utils import set_seed

def select_future(trajectories, scores):
    # [1, 2, 6, 80, 4]& [1, 6]
    trajectories = trajectories.squeeze(0) # [2, 6, 80, 4]
    # scores = scores.squeeze(0)
    best_mode = torch.argmax(scores, dim=-1) # [1, 6] # [1]
    best_mode_future = trajectories[torch.arange(trajectories.shape[0]), best_mode, :, :2] # [2, 80, 2]
    # best_mode_future = best_mode_future.squeeze(1) 
    return best_mode_future

class OpenLoopTestProcessor(DataProcess):
    def __init__(self):
        self.num_neighbors = 1
        self.hist_len = 11
        self.future_len = 80
        self.n_lanes = 6
        self.n_crosswalks = 4
        self.n_refline_waypoints = 1000
        self.point_dir = ''

    def build_map_forplot(self, map_features, dynamic_map_states):
        lanes = {}
        roads = {}
        road_edges = {}
        road_lines = {}
        stop_signs = {}
        crosswalks = {}
        speed_bumps = {}

        # static map features
        for map in map_features:
            map_type = map.WhichOneof("feature_data")
            map_id = map.id
            map = getattr(map, map_type)

            if map_type == 'lane':
                lanes[map_id] = map
            elif map_type == 'road_line':
                road_lines[map_id] = map
            elif map_type == 'road_edge':
                road_edges[map_id] = map
            elif map_type == 'stop_sign':
                stop_signs[map_id] = map
            elif map_type == 'crosswalk': 
                crosswalks[map_id] = map
            elif map_type == 'speed_bump':
                speed_bumps[map_id] = map
            else:
                raise TypeError

        roads.update(road_edges)
        roads.update(road_lines)

        # dynamic map features
        traffic_signals = dynamic_map_states

        # all map features
        map = {"lane": lanes, "road_line": road_lines, "road_edge": road_edges, "crosswalk": crosswalks, 
                    "speed_bump": speed_bumps, "stop_sign": stop_signs, "dynamic_map_states": traffic_signals}
        return map
        
    def process_frame(self, parsed_data): # , test=True
        self.scenario_id = parsed_data.scenario_id
        objects_of_interest = parsed_data.objects_of_interest

        tracks_to_predict = parsed_data.tracks_to_predict
        id_list = {}
        tracks_list = []
        for ids in tracks_to_predict:
            id_list[parsed_data.tracks[ids.track_index].id] = ids.track_index
            tracks_list.append(ids.track_index)
        interact_list = []
        for int_id in objects_of_interest:
            interact_list.append(id_list[int_id])

        self.build_map(parsed_data.map_features, parsed_data.dynamic_map_states)

        # if test:
        #     if parsed_data.tracks[tracks_to_predict[0].track_index].object_type==1:
        #         self.sdc_ids_list = [([tracks_list[0], tracks_list[1]],1)]
        #     else:
        #         self.sdc_ids_list = [(tracks_list,1)] 
        # else:
        self.interactive_process(tracks_list, interact_list, parsed_data.tracks)

        # for pairs in self.sdc_ids_list:
        pairs = self.sdc_ids_list[0]
        sdc_ids, interesting = pairs[0], pairs[1]                   
        # process data
        ego = self.ego_process(sdc_ids, parsed_data.tracks)

        ego_type = parsed_data.tracks[sdc_ids[0]].object_type
        neighbor_type = parsed_data.tracks[sdc_ids[1]].object_type
        object_type = np.array([ego_type, neighbor_type])
        self.object_type = object_type
        ego_index = parsed_data.tracks[sdc_ids[0]].id
        neighbor_index = parsed_data.tracks[sdc_ids[1]].id
        object_index = np.array([ego_index, neighbor_index])

        neighbors, neighbors_to_predict = self.neighbors_process(sdc_ids, parsed_data.tracks)
        map_lanes = np.zeros(shape=(2, 6, 300, 17), dtype=np.float32)
        map_crosswalks = np.zeros(shape=(2, 4, 100, 3), dtype=np.float32)
        map_lanes[0], map_crosswalks[0] = self.map_process(ego[0])
        map_lanes[1], map_crosswalks[1] = self.map_process(ego[1])

        # if test:
        #     ground_truth = np.zeros((2, self.future_len, 5))
        # else:
        ground_truth = self.ground_truth_process(sdc_ids, parsed_data.tracks)
        ego, neighbors, map_lanes, map_crosswalks, ground_truth, region_dict = self.normalize_data(ego, neighbors, map_lanes, map_crosswalks, ground_truth)

        neighbors = np.concatenate([ego[1][np.newaxis,...], neighbors], axis=0)
        obs = {'ego_state': ego[0], 'neighbors_state':neighbors, 'map_lanes': map_lanes[:, :,:200:2], 
               'map_crosswalks': map_crosswalks[:,:, :100:2]}       
        
        return obs, sdc_ids[0], sdc_ids[1], ground_truth


def open_loop_test():
    # logging
    save_path = f"/visualization/{args.name}/"
    print("Use device: {}".format(args.device))
    # """load from dir path"""
    test_files = glob.glob(args.test_set+'/*')

    # data processor
    processor = OpenLoopTestProcessor()
    # seed
    set_seed(args.seed)
    from model.JAM import JAM
    jam_model = JAM(modalities=6,class_query_topK=1,encoder_layers=6,future_len=80,neighbors_to_predict=1,class_query_N=64).to(args.device)
    # load param
    model_ckpts = load_model_from_ddp_ckpts(args.jam_model_path, args.device)
    jam_model.load_state_dict(model_ckpts)
    jam_model.eval()
    total_params = sum(p.numel() for p in jam_model.parameters())
    print(f"jam Total parameters: {total_params}")
    
    max_N = args.max_N
    for model_name, model in zip(["jam_model"],[jam_model]):
        # logging.info(f"{model_name}".center(80,"="))
        # param init
        test_id = 0
        for file in test_files:
            # load scene
            scenarios = tf.data.TFRecordDataset(file)
            if test_id>max_N:
                break
            # iterate thru scenarios
            for scenario in scenarios:
                parsed_data = scenario_pb2.Scenario()
                parsed_data.ParseFromString(scenario.numpy())
                scenario_id = parsed_data.scenario_id
                
                test_id += 1
                if test_id>max_N:
                    break
                print(f"\r model: {model_name} | count_N: {test_id}/{max_N} | Testing scenario: {scenario_id}", end=" ")

                """model forward"""
                curr_t = 10
                data = processor.process_frame(parsed_data)
                if data is None:
                    continue
                else:
                    obs, sdc_ids, neighbor_ids, gt_future = data
                # prepare data
                inputs = {
                    'ego_state': torch.from_numpy(obs['ego_state']).unsqueeze(0).to(args.device),
                    'neighbors_state': torch.from_numpy(obs['neighbors_state']).unsqueeze(0).to(args.device),
                    'map_lanes': torch.from_numpy(obs['map_lanes']).unsqueeze(0).to(args.device),
                    'map_crosswalks': torch.from_numpy(obs['map_crosswalks']).unsqueeze(0).to(args.device)
                }
                # level-k reasoning
                with torch.no_grad():
                    outputs = model(inputs)
                trajectories_multimodal = outputs[f'joint_interactions']
                scores_multimodal = outputs[f'joint_scores']
                # [1, 2, 6, 80, 4]& [1, 2, 6]
                scores_multimodal = scores_multimodal.mean(dim=1)
                scores_multimodal = F.softmax(scores_multimodal,dim=-1)
                # [1, 2, 6, 80, 4]& [1, 6]
                """plot"""
                if args.render:
                    gt_future_global = transform_to_global_frame(curr_t, gt_future, 
                                                            processor.current_xyzh[0], [neighbor_ids], parsed_data.tracks)
                    traj_multimodal_global_list = transform_to_global_frame_multimodal(curr_t, trajectories_multimodal.cpu().numpy(), 
                                                            processor.current_xyzh[0], [neighbor_ids], parsed_data.tracks)  # [2, 6, 80, 4] > list[2,6,[80,4]]
                    map = processor.build_map_forplot(parsed_data.map_features, parsed_data.dynamic_map_states)
                    plot_scenario(curr_t, sdc_ids, [neighbor_ids], map, processor.current_xyzh[0], 
                                parsed_data.tracks, gt_future_global, traj_multimodal_global_list,scores_multimodal.cpu().numpy(),args.name, model_name, scenario_id, save_path=save_path, save=args.render)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Open-loop Testing')
    parser.add_argument('--name', type=str, help='log name (default: "jam_visual_res")',default="jam_visual_res")
    parser.add_argument('--test_set', type=str, help='path to testing datasets',default="/Jam/waymo_dataset_1_2/validation_interactive")
    parser.add_argument('--jam_model_path', type=str, help='path to saved model',default="/Jam/jam_log/jam/epochs_29.pth")
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--max_N', type=int, help='the number of plot', default=1000)
    parser.add_argument('--render', action="store_true", help='if render the scenario', default=True)
    parser.add_argument('--device', type=str, help='run on which device', default='cuda')
    args = parser.parse_args()

    open_loop_test()
