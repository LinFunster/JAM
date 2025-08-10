import os
import torch
import logging
import glob
import tensorflow as tf
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
from google.protobuf import text_format

from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2

import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random


def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())

def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DrivingData(Dataset):
    def __init__(self, data_dir, test_set=False):
        self.data_list = glob.glob(data_dir)
        self.test_set = test_set

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data['ego'][0]
        neighbor = np.concatenate([data['ego'][1][np.newaxis,...], data['neighbors']], axis=0)

        map_lanes = data['map_lanes'][:, :, :200:2]
        map_crosswalks = data['map_crosswalks'][:, :, :100:2]
        if self.test_set:
            scene_id = self.data_list[idx].split('/')[-1].split('_')[0]
            object_index = data['object_index']
            current_state = data['current_state']

            return ego, neighbor, map_lanes, map_crosswalks, object_index, scene_id, current_state
        else:
            ego_future_states = data['gt_future_states'][0]
            neighbor_future_states = data['gt_future_states'][1]
            object_type = data['object_type']
        
            return ego, neighbor, map_lanes, map_crosswalks, ego_future_states, neighbor_future_states, object_type
        
def joint_gmm_loss(trajectories, convs, probs, ground_truth):
    metric = [29, 49, 79]
    distance = torch.norm(trajectories[:, :, :, : ,:2] - ground_truth[:, :, None, :, :2], dim=-1)
    ndistance = distance[...,metric].sum(-1) + distance.mean(-1) 
    best_mode = torch.argmin(ndistance.mean(1), dim=-1) 
    B, N = trajectories.shape[0], trajectories.shape[1]
    
    #[b,n,t,2]
    best_mode_future = trajectories[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, None, None]].squeeze(2)
    #[b,n,t,3]
    convs = convs[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, None, None]].squeeze(2)

    dx = best_mode_future[...,0] - ground_truth[...,0]
    dy = best_mode_future[...,1] - ground_truth[...,1]

    log_std_x = torch.clip(convs[...,0], 0, 5)
    log_std_y = torch.clip(convs[...,1], 0, 5)

    std_x, std_y = torch.exp(log_std_x), torch.exp(log_std_y)

    reg_gmm_log_coefficient = log_std_x + log_std_y  # (batch_size, num_timestamps)
    reg_gmm_exp = 0.5  * ((dx**2) / (std_x**2) + (dy**2) / (std_y**2))
    loss = reg_gmm_log_coefficient + reg_gmm_exp
    loss = loss.mean(-1) + loss[..., metric].sum(-1)

    prob_loss = F.cross_entropy(input=probs, target=best_mode, label_smoothing=0.2)
    loss = loss + 2*prob_loss
    loss = loss.mean()

    return loss, best_mode, best_mode_future, convs

def marginal_gmm_loss(trajectories, convs, probs, ground_truth, class_query_N):
    metric = [29, 49, 79]
    distance = torch.norm(trajectories[:, :, :, : ,:2] - ground_truth[:, :, None, :, :2], dim=-1)
    ndistance = distance[...,metric].sum(-1) + distance.mean(-1) # [B, N, M*Class_N]
    # get class best idx
    # best_mode = judge_action_type_batch(ground_truth) # [B, N] behavior-class 
    best_mode = get_intention_torch(ground_truth).to(ground_truth.device) # [B, N] kmeans-class 
    # valid class
    valid_mask = best_mode!=-1 # [B, N]
    class_ndistance = ndistance.view(*ndistance.shape[:2],class_query_N,-1) # [B, N, Class_N, M]
    B,N,Class_N,M = class_ndistance.shape
    class_local_best_mode_idx = torch.argmin(class_ndistance, dim=-1) # [B, N, Class_N]
    valid_local_best_mode_idx = class_local_best_mode_idx[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None],best_mode[:, :, None]].squeeze(2) # [B, N]
    best_mode[valid_mask] = best_mode[valid_mask]*M + valid_local_best_mode_idx[valid_mask]
    # invalid class
    invalid_global_best_mode_idx = torch.argmin(ndistance, dim=-1) # [B, N]
    invalid_mask = best_mode==-1 # [B, N]
    best_mode[invalid_mask] = invalid_global_best_mode_idx[invalid_mask] # [B, N]

    #[b,n,t,2]
    best_mode_future = trajectories[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, :, None]].squeeze(2)
    #[b,n,t,3]
    convs = convs[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, :, None]].squeeze(2)

    dx = best_mode_future[...,0] - ground_truth[...,0]
    dy = best_mode_future[...,1] - ground_truth[...,1]

    log_std_x = torch.clip(convs[...,0], 0, 5)
    log_std_y = torch.clip(convs[...,1], 0, 5)

    std_x, std_y = torch.exp(log_std_x), torch.exp(log_std_y)

    reg_gmm_log_coefficient = log_std_x + log_std_y  # (batch_size, num_timestamps)
    reg_gmm_exp = 0.5  * ((dx**2) / (std_x**2) + (dy**2) / (std_y**2))
    loss = reg_gmm_log_coefficient + reg_gmm_exp
    loss = loss.mean(-1) + loss[..., metric].sum(-1)

    prob_loss = F.cross_entropy(input=probs.permute(0, 2, 1), target=best_mode.detach(), label_smoothing=0.2)
    loss = loss + 2*prob_loss
    loss = loss.mean()

    return loss, best_mode, best_mode_future, convs

def marginal_anchor_free_gmm_loss(trajectories, convs, probs, ground_truth, class_query_N):
    metric = [29, 49, 79]
    distance = torch.norm(trajectories[:, :, :, : ,:2] - ground_truth[:, :, None, :, :2], dim=-1)
    ndistance = distance[...,metric].sum(-1) + distance.mean(-1) # [B, N, M*Class_N]
    # get class best idx
    ndistance_shape = ndistance.view(*ndistance.shape[:2],class_query_N,-1) # [B, N, Class_N, M]
    B,N,Class_N,M = ndistance_shape.shape
    # valid class
    best_mode = torch.argmin(ndistance, dim=-1) # [B, N]
    #[b,n,t,2]
    best_mode_future = trajectories[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, :, None]].squeeze(2)
    #[b,n,t,3]
    convs = convs[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], best_mode[:, :, None]].squeeze(2)

    dx = best_mode_future[...,0] - ground_truth[...,0]
    dy = best_mode_future[...,1] - ground_truth[...,1]

    log_std_x = torch.clip(convs[...,0], 0, 5)
    log_std_y = torch.clip(convs[...,1], 0, 5)

    std_x, std_y = torch.exp(log_std_x), torch.exp(log_std_y)

    reg_gmm_log_coefficient = log_std_x + log_std_y  # (batch_size, num_timestamps)
    reg_gmm_exp = 0.5  * ((dx**2) / (std_x**2) + (dy**2) / (std_y**2))
    loss = reg_gmm_log_coefficient + reg_gmm_exp
    loss = loss.mean(-1) + loss[..., metric].sum(-1)

    prob_loss = F.cross_entropy(input=probs.permute(0, 2, 1), target=best_mode.detach(), label_smoothing=0.2)
    loss = loss + 2*prob_loss
    loss = loss.mean()

    return loss, best_mode, best_mode_future, convs

def get_intention_torch(ground_truth_batch): # ground_truth_batch [B,N,T,F]
    object_type = ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
    with open("utils/class_query_waymo_data/cluster_64_center_dict.pkl", 'rb') as f:
        intention_points_dict = pickle.load(f)
    intention_points = []
    for cur_type in object_type:
        cur_intention_points = intention_points_dict[cur_type] 
        cur_intention_points = torch.from_numpy(cur_intention_points).float().view(1,-1,2).cuda() # [1,64,2]
        intention_points.append(cur_intention_points) # cur_intention_points torch.Size([1, 64, 2])

    # Unset=0(unvalid type), Vehicle=1, Pedestrian=2, Cyclist=3, Other=4
    ground_truth_type = ground_truth_batch[...,0,-1] # [B,N]
    ground_truth_type_valid = torch.zeros_like(ground_truth_type).bool().cuda() # [B,N]
    target_intention_points = torch.zeros((*ground_truth_type.shape,*cur_intention_points.shape[1:])).cuda() # [B,N,64,2]
    for type_i in range(1, 4):
        target_intention_points[ground_truth_type[..., -1]==type_i] = intention_points[type_i-1][0]
        ground_truth_type_valid[ground_truth_type==type_i] = True

    # mask
    gt_valid_mask = torch.ne(ground_truth_batch[..., :2].sum(-1), 0) # [B,N,T] 
    B,N,_ = gt_valid_mask.shape
    gt_final_valid_idx = torch.zeros(gt_valid_mask.shape[:2]).long() # torch.Size([64, 2])
    for k in range(gt_valid_mask.shape[2]):
        cur_valid_mask = gt_valid_mask[:, :, k] > 0  # [B,N] torch.Size([64, 2])
        gt_final_valid_idx[cur_valid_mask] = k # 

    gt_valid = ground_truth_batch[torch.arange(B)[:,None,None],torch.arange(N)[None,:,None],gt_final_valid_idx[:,:,None], :2] # [B,N,1,2]
    # best mode
    dist = (gt_valid - target_intention_points).norm(dim=-1)  # (B, N, 1, 2) - [B,N,64,2] > [B,N,64]
    gt_positive_idx = dist.argmin(dim=-1)  # [B,N]
    gt_positive_idx[~ground_truth_type_valid] = -1
    return gt_positive_idx

def judge_action_type_batch(ground_truth):
    action_type_dict =  {'stationary':0, 'straight':1, 'straight_right':2, 
                        'straight_left':3, 'right_turn':4, 'right_u_turn':5, 
                        'left_turn':6, 'left_u_turn':7, 'None':-1} 
    # traj = np.concatenate([ego[:, :5], ground_truth[0, :, :5]], axis=0)
    traj = ground_truth[..., :5] # [B,N,T,5]
    valid = traj[..., -1] != 0 # True is valid [B,N,T]
    future_xy = traj[..., :2] # [B,N,T,2]
    future_yaw = traj[..., 2] # [B,N,T]
    future_speed = traj[..., 3:] # [B,N,T,2]
    future_speed = torch.norm(future_speed, dim=-1) # [B,N,T]

    kMaxSpeedForStationary = 2.0                 # (m/s) 2.0
    kMaxDisplacementForStationary = 5.0          # (m) 5.0
    kMaxLateralDisplacementForStraight = 5.0     # (m) 5.0
    kMinLongitudinalDisplacementForUTurn = -5.0  # (m) -5.0
    kMaxAbsHeadingDiffForStraight = torch.pi / 6.0   # (rad) np.pi / 6.0
    
    # get first and last index
    B, N, T, F = ground_truth.shape
    first_valid_index = torch.zeros((B, N)).long().to(ground_truth.device)  # [B,N]
    last_valid_index = torch.zeros((B, N)).long().to(ground_truth.device)  # [B,N]
    for k in range(1, T):
        valid_mask = valid[..., k]  # [B,N]
        last_valid_index[valid_mask] = k  # [B,N]

    # final_displacement
    future_xy_last = future_xy[torch.arange(B)[:,None],torch.arange(N)[None,:],last_valid_index] # [B,N,T,2] > [B,N,2]
    future_xy_first = future_xy[torch.arange(B)[:,None],torch.arange(N)[None,:],first_valid_index] # [B,N,T,2] > [B,N,2]
    xy_delta = future_xy_last-future_xy_first # [B,N,2]
    final_displacement = torch.norm(xy_delta,dim=-1) # [B,N]

    # heading_delta
    future_yaw_last = future_yaw[torch.arange(B)[:,None],torch.arange(N)[None,:],last_valid_index] # [B,N,T] > [B,N]
    future_yaw_first = future_yaw[torch.arange(B)[:,None],torch.arange(N)[None,:],first_valid_index] # [B,N,T] > [B,N]
    heading_delta = future_yaw_last-future_yaw_first # [B,N]

    # max_speed
    future_speed[valid==False] = 0 # [B,N,T]
    max_speed = torch.max(future_speed,dim=-1).values # [B,N]

    # param init
    action_type = torch.zeros((B, N)).long().to(ground_truth.device)  # [B,N]
    is_flag = torch.zeros((B, N)).bool().to(ground_truth.device)  # [B,N]
    """invalid type"""
    # invalid type
    first_valid = valid[torch.arange(B)[:,None],torch.arange(N)[None,:],first_valid_index] # [B,N,T,5] > [B,N]
    mask = torch.logical_or(first_valid == False, last_valid_index==0) # [B,N]
    action_type[mask] = -1
    is_flag[mask] = True # [B,N]
    """stationary"""
    # stationary [B,N]
    stationary = torch.logical_and(max_speed < kMaxSpeedForStationary, final_displacement < kMaxDisplacementForStationary)
    stationary = torch.logical_and(stationary,is_flag==False)
    action_type[stationary] = action_type_dict["stationary"]
    is_flag[stationary] = True 
    """straight"""
    # straight
    straight_base = torch.abs(heading_delta) < kMaxAbsHeadingDiffForStraight
    straight_base = torch.logical_and(straight_base,is_flag==False)
    straight = torch.logical_and(straight_base, torch.abs(xy_delta[...,1]) < kMaxLateralDisplacementForStraight)
    action_type[straight] = action_type_dict["straight"]
    is_flag[straight] = True 
    # straight_right
    straight_base = torch.logical_and(straight_base,is_flag==False)
    straight_right = torch.logical_and(straight_base, xy_delta[...,1] < 0)
    action_type[straight_right] = action_type_dict["straight_right"]
    is_flag[straight_right] = True 
    # straight_left
    straight_left = torch.logical_and(straight_base,is_flag==False)
    action_type[straight_left] = action_type_dict["straight_left"]
    is_flag[straight_left] = True 
    """right_turn"""
    # right_u_turn
    right_turn_base = torch.logical_and(heading_delta < -kMaxAbsHeadingDiffForStraight, xy_delta[...,1])
    right_turn_base = torch.logical_and(right_turn_base,is_flag==False)
    right_u_turn = torch.logical_and(right_turn_base,xy_delta[...,0] < kMinLongitudinalDisplacementForUTurn)
    action_type[right_u_turn] = action_type_dict["right_u_turn"]
    is_flag[right_u_turn] = True
    # right_turn
    right_turn = torch.logical_and(right_turn_base,is_flag==False)
    action_type[right_turn] = action_type_dict["right_turn"]
    is_flag[right_turn] = True
    """left_turn"""
    # left_u_turn
    left_turn_base = torch.logical_and(heading_delta > kMaxAbsHeadingDiffForStraight, (-xy_delta[...,1]))
    left_turn_base = torch.logical_and(left_turn_base,is_flag==False)
    left_u_turn = torch.logical_and(right_turn_base,xy_delta[...,0] < kMinLongitudinalDisplacementForUTurn)
    action_type[left_u_turn] = action_type_dict["left_u_turn"]
    is_flag[left_u_turn] = True
    # left_turn
    left_turn = torch.logical_and(left_turn_base,is_flag==False)
    action_type[left_turn] = action_type_dict["left_turn"]
    is_flag[left_turn] = True
    """no_type"""
    no_type = is_flag==False
    action_type[no_type] = action_type_dict["None"]
    is_flag[no_type] = True 
    return action_type # [B,N]

def jam_loss(outputs, ego_future, neighbor_future, class_query_N):
    loss: torch.tensor = 0
    neighbor_future_valid = torch.ne(neighbor_future[..., :2].sum(-1), 0)
    ego_future_valid = torch.ne(ego_future[..., :2].sum(-1), 0)
    gt_future = torch.stack([ego_future, neighbor_future], dim=1)
    # marginal gmm loss
    trajectories = outputs[f'marginal_interactions'][..., :2]
    scores = outputs[f'marginal_scores']
    convs = outputs[f'marginal_interactions'][..., 2:]
    ego = trajectories[:, 0] * ego_future_valid.unsqueeze(1).unsqueeze(-1)
    neighbor = trajectories[:, 1] * neighbor_future_valid.unsqueeze(1).unsqueeze(-1)
    trajectories = torch.stack([ego, neighbor], dim=1)
    gloss, best_mode, future, _ = marginal_gmm_loss(trajectories, convs, scores, gt_future, class_query_N)
    loss += gloss
    # joint gmm loss
    trajectories = outputs[f'joint_interactions'][..., :2]
    scores = outputs[f'joint_scores']
    ego = trajectories[:, 0] * ego_future_valid.unsqueeze(1).unsqueeze(-1)
    neighbor = trajectories[:, 1] * neighbor_future_valid.unsqueeze(1).unsqueeze(-1)
    trajectories = torch.stack([ego, neighbor], dim=1)
    convs = outputs[f'joint_interactions'][..., 2:]
    gloss, best_mode, future, _ = joint_gmm_loss(trajectories, convs, scores.sum(1), gt_future)
    loss += gloss
    return loss, (future,best_mode,scores)

def motion_metrics(trajectories, ego_future, neighbor_future):
    ego_future_valid = torch.ne(ego_future[..., :2], 0)
    ego_trajectory = trajectories[:, 0] * ego_future_valid
    distance = torch.norm(ego_trajectory[:, 4::5, :2] - ego_future[:, 4::5, :2], dim=-1)
    egoADE = torch.mean(distance)
    egoFDE = torch.mean(distance[:, -1])

    neigbhor_future_valid = torch.ne(neighbor_future[..., :2], 0)
    neighbor_trajectory = trajectories[:, 1] * neigbhor_future_valid
    distance = torch.norm(neighbor_trajectory[:, 4::5, :2] - neighbor_future[:, 4::5, :2], dim=-1)
    neighborADE = torch.mean(distance)
    neighborFDE = torch.mean(distance[:, -1])

    return egoADE.item(), egoFDE.item(), neighborADE.item(), neighborFDE.item()

# step_configurations
# // The prediction samples and parameters used to compute metrics at a specific
# // time step. Time in seconds can be computed as (measurement_step + 1) /
# // prediction_steps_per_second. Metrics are computed for each step in the list
# // as if the given measurement_step were the last step in the predicted
# // trajectory.
# Define metrics to measure the prediction
def default_metrics_config():
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
        track_steps_per_second: 10
        prediction_steps_per_second: 2
        track_history_samples: 10
        track_future_samples: 80
        speed_lower_bound: 1.4
        speed_upper_bound: 11.0
        speed_scale_lower: 0.5
        speed_scale_upper: 1.0
        step_configurations {
            measurement_step: 5
            lateral_miss_threshold: 1.0
            longitudinal_miss_threshold: 2.0
        }
        step_configurations {
            measurement_step: 9
            lateral_miss_threshold: 1.8
            longitudinal_miss_threshold: 3.6
        }
        step_configurations {
            measurement_step: 15
            lateral_miss_threshold: 3.0
            longitudinal_miss_threshold: 6.0
        }
        max_predictions: 6
    """
    text_format.Parse(config_text, config)

    return config

class MotionMetrics:
    """Wrapper for motion metrics computation."""
    def __init__(self):
        super().__init__()
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_trajectory = []
        self._prediction_score = []
        self._object_type = []
        self._metrics_config = default_metrics_config()

    def reset_states(self):
        self._ground_truth_trajectory = []
        self._ground_truth_is_valid = []
        self._prediction_trajectory = []
        self._prediction_score = []
        self._object_type = []

    def update_state(self, prediction_trajectory, prediction_score, ground_truth_trajectory, ground_truth_is_valid, object_type):
        self._prediction_trajectory.append(prediction_trajectory[..., 4::5,:].clone().detach().cpu())
        self._prediction_score.append(prediction_score.clone().detach().cpu())
        self._ground_truth_trajectory.append(ground_truth_trajectory.cpu())
        self._ground_truth_is_valid.append(ground_truth_is_valid[..., -1].cpu())
        self._object_type.append(object_type.cpu())

    def result(self):
        # [batch_size, 1, top_k, 2, steps, 2].
        prediction_trajectory = torch.cat(self._prediction_trajectory, dim=0)
        # [batch_size, 1, top_k].
        prediction_score = torch.cat(self._prediction_score, dim=0)
        # [batch_size, 1, 2, gt_steps, 7].
        ground_truth_trajectory = torch.cat(self._ground_truth_trajectory, dim=0)
        # [batch_size, 1, gt_steps].
        ground_truth_is_valid = torch.cat(self._ground_truth_is_valid, dim=0)
        # [batch_size, 1].
        object_type = torch.cat(self._object_type, dim=0)

        # We are predicting more steps than needed by the eval code. Subsample.
        #interval = (self._metrics_config.track_steps_per_second // self._metrics_config.prediction_steps_per_second)
        # Prepare these into shapes expected by the metrics computation.
        # [batch_size, top_k, num_agents_per_joint_prediction, pred_steps, 2].
        # num_agents_per_joint_prediction is 1 here.
        if len(prediction_trajectory.shape)>=4:
            prediction_trajectory = prediction_trajectory.unsqueeze(dim=1).numpy()
            prediction_score = prediction_score.unsqueeze(dim=1).numpy()
        else:
            prediction_trajectory = prediction_trajectory.numpy()

        # [batch_size, num_agents_per_joint_prediction, gt_steps, 7].

        ground_truth_trajectory = ground_truth_trajectory.numpy()
        # [batch_size, num_agents_per_joint_prediction, gt_steps].
        ground_truth_is_valid = ground_truth_is_valid.numpy()

        # [batch_size, num_agents_per_joint_prediction].
        object_type = object_type.numpy()
        b = ground_truth_trajectory.shape[0]

        prediction_ground_truth_indices = tf.cast(tf.concat([tf.zeros((b, 1, 1)), tf.ones((b, 1, 1))],axis=-1),tf.int64)
        # print(object_type.shape)
        ground_truth_is_valid = tf.convert_to_tensor(ground_truth_is_valid)
        prediction_ground_truth_indices_mask = tf.ones_like(prediction_ground_truth_indices, dtype=tf.float32)
        valid_gt_all = tf.cast(tf.math.greater_equal(tf.reduce_sum(tf.cast(ground_truth_is_valid,tf.float32), axis=-1), 1), tf.float32)
        valid_gt_all = valid_gt_all[:, tf.newaxis, :] * prediction_ground_truth_indices_mask
        valid_gt_all = tf.cast(valid_gt_all, tf.bool)

        metric_values = py_metrics_ops.motion_metrics(
                config=self._metrics_config.SerializeToString(),
                prediction_trajectory=tf.convert_to_tensor(prediction_trajectory), # [414,1,6,2,16,2]
                prediction_score=tf.convert_to_tensor(prediction_score),
                ground_truth_trajectory=tf.convert_to_tensor(ground_truth_trajectory), # [414,2,91,7]
                ground_truth_is_valid=ground_truth_is_valid,
                object_type=tf.convert_to_tensor(object_type),
                prediction_ground_truth_indices=prediction_ground_truth_indices,
                prediction_ground_truth_indices_mask=valid_gt_all)


        metric_names = config_util.get_breakdown_names_from_motion_config(self._metrics_config)
        results = {}

        for i, m in enumerate(['minADE', 'minFDE', 'miss_rate', 'overlap_rate', 'mAP']):
            for j, n in enumerate(metric_names):
                results[f'{m}_{n}'] = metric_values[i][j].numpy()

        return results
