import os
import torch
import numpy as np
import matplotlib as mpl
from scipy import signal
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.affinity import affine_transform, rotate
from collections import OrderedDict

def load_model_from_ddp_ckpts(model_path, device):
    model_ckpts = torch.load(model_path, map_location=device)['model_states']
    model_ckpts = remove_prefix(model_ckpts)
    return model_ckpts

def remove_prefix(storage):
    new_state_dict = OrderedDict()
    for k, v in storage.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    return new_state_dict

def transform_to_global_frame(timestep, trajectories, ego_pose, predict_ids, tracks):
    # trajectories [2,80,2]
    trajectories = trajectories[...,:2] # [2,6,80,2]
    ego_p = ego_pose[:2]
    ego_h = ego_pose[3]

    global_trajectories = []   
    line = LineString(trajectories[0])
    line = rotate(line, ego_h, origin=(0, 0), use_radians=True)
    line = affine_transform(line, [1, 0, 0, 1, ego_p[0], ego_p[1]])
    line = np.array(line.coords)
    traj = np.insert(line, 0, ego_p, axis=0)
    traj = trajectory_smoothing(traj)
    global_trajectories.append(traj)

    for i, j in enumerate(predict_ids):
        current_state = np.array([tracks[j].states[timestep].center_x, tracks[j].states[timestep].center_y])
        line = LineString(trajectories[i+1])
        line = rotate(line, ego_h, origin=(0, 0), use_radians=True)
        line = affine_transform(line, [1, 0, 0, 1, ego_p[0], ego_p[1]])
        line = np.array(line.coords)
        traj = np.insert(line, 0, current_state, axis=0)
        traj = trajectory_smoothing(traj)
        global_trajectories.append(traj)

    return global_trajectories

def transform_to_global_frame_multimodal(timestep, trajectories, ego_pose, predict_ids, tracks):
    # trajectories  
    trajectories = trajectories[0,...,:2] # [2,6,80,2]
    ego_p = ego_pose[:2]
    ego_h = ego_pose[3]
    modal_N = trajectories.shape[1]
    
    global_trajectories = []
    multimodal_trajs = []
    for modal_i in range(modal_N):
        line = LineString(trajectories[0,modal_i])
        line = rotate(line, ego_h, origin=(0, 0), use_radians=True)
        line = affine_transform(line, [1, 0, 0, 1, ego_p[0], ego_p[1]])
        line = np.array(line.coords)
        traj = np.insert(line, 0, ego_p, axis=0)
        traj = trajectory_smoothing(traj)
        multimodal_trajs.append(traj)
    global_trajectories.append(multimodal_trajs)

    for i, j in enumerate(predict_ids):
        current_state = np.array([tracks[j].states[timestep].center_x, tracks[j].states[timestep].center_y])
        multimodal_trajs = []
        for modal_i in range(modal_N):
            line = LineString(trajectories[i+1,modal_i])
            line = rotate(line, ego_h, origin=(0, 0), use_radians=True)
            line = affine_transform(line, [1, 0, 0, 1, ego_p[0], ego_p[1]])
            line = np.array(line.coords)
            traj = np.insert(line, 0, current_state, axis=0)
            traj = trajectory_smoothing(traj)
            multimodal_trajs.append(traj)
        global_trajectories.append(multimodal_trajs)

    return np.array(global_trajectories)

def trajectory_smoothing(trajectory):
    x = trajectory[:,0]
    y = trajectory[:,1]

    window_length = 25
    x = signal.savgol_filter(x, window_length=window_length, polyorder=3)
    y = signal.savgol_filter(y, window_length=window_length, polyorder=3)
   
    return np.column_stack([x, y])


def plot_scenario(timestep, sdc_id, predict_ids, map_features, ego_pose, agents, gt_traj, trajectories, scores, name, model_name, scenario_id, save_path=None, save=False):
    plt.ion()
    fig = plt.gcf()
    dpi = 600
    size_inches = 4800 / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_tight_layout(True)

    _plot_map_features(map_features)
    _plot_traffic_signals(map_features['dynamic_map_states'][timestep])
    _plot_gt_trajectories(gt_traj)
    _plot_trajectories_multimodal(trajectories, scores)
    _plot_agents(agents, timestep, sdc_id, predict_ids)

    plt.gca().set_facecolor('silver')
    plt.gca().margins(0)
    plt.gca().set_aspect('equal')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axis([-60 + ego_pose[0], 60 + ego_pose[0], -60 + ego_pose[1], 60 + ego_pose[1]])

    if save:
        if save_path is None:
            save_path = f"./olt_log/{name}/{model_name}"
        save_path = save_path + f"{model_name}"
        os.makedirs(save_path, exist_ok=True)
        # plt.savefig(f'{save_path}/{scenario_id}_{timestep}.svg')
        plt.savefig(f'{save_path}/{scenario_id}_{timestep}.png',dpi=dpi)
    else:
        plt.pause(1)

    plt.clf()


def _plot_agents(tracks, timestep, sdc_id, predict_ids):
    for id, track in enumerate(tracks):
        if not track.states[timestep].valid:
            continue
        else:
            state = track.states[timestep]
            pos_x, pos_y = state.center_x, state.center_y
            length, width = state.length, state.width

            if id in predict_ids:
                color = 'xkcd:blue'
            elif id == sdc_id:
                color = 'xkcd:red' 
            else:
                color = 'xkcd:olive'

            rect = plt.Rectangle((pos_x - length/2, pos_y - width/2), length, width, linewidth=2, color=color, alpha=0.9, zorder=3,
                                  transform=mpl.transforms.Affine2D().rotate_around(*(pos_x, pos_y), state.heading) + plt.gca().transData)
            plt.gca().add_patch(rect)

def _plot_gt_trajectories(trajectories):
    traj = trajectories[0]
    for i, traj in enumerate(trajectories):
        plt.plot(traj[:, 0], traj[:, 1], linewidth=1.5, color="black", linestyle="--", zorder=3)

def _plot_trajectories_multimodal(trajectories, scores):
    scores = scores[0]  # [6]
    
    # Define zorder mapping based on scores
    zorder_mapping = {0: 2, 1: 2, 2: 2, 3: 1, 4: 1, 5: 1}  # example, you can extend this
    idx_sort = np.argsort(-scores)  # Sort scores in descending order
    
    for agent_i, agent_trajs in enumerate(trajectories):
        for modal_i, traj in enumerate(agent_trajs):  # 80, 2
            if agent_i == 0:
                cmap = plt.get_cmap('autumn_r')  # Reverse the 'autumn' colormap
                # Map scores[modal_i] to color
                color = cmap((scores[modal_i] - min(scores)) / (max(scores) - min(scores)))
                
                # Set zorder based on score
                zorder_sort = zorder_mapping.get(np.where(idx_sort == modal_i)[0][0], 3)  # Default to 3 if score is out of expected range
                
                # Plot trajectory with scatter
                plt.plot(traj[:, 0], traj[:, 1], linewidth=2, color=color, alpha=0.6, zorder=zorder_sort)
                # plt.scatter(traj[:, 0], traj[:, 1], color=color, s=3, alpha=0.5, zorder=zorder_sort)
            else:
                cmap = plt.get_cmap('winter_r')  # Reverse the 'winter' colormap
                # Map scores[modal_i] to color
                color = cmap((scores[modal_i] - min(scores)) / (max(scores) - min(scores)))
                
                # Set zorder based on score
                zorder_sort = zorder_mapping.get(np.where(idx_sort == modal_i)[0][0], 3)
                
                # Plot trajectory with scatter
                plt.plot(traj[:, 0], traj[:, 1], linewidth=2, color=color, alpha=0.6, zorder=zorder_sort)
                # plt.scatter(traj[:, 0], traj[:, 1], color=color, s=3, alpha=0.5, zorder=zorder_sort)
        
        # Add a more compact colorbar with tighter spacing and aspect ratio adjustment
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(scores), vmax=max(scores)))
        sm.set_array([])  # Must set an empty array
        
        if agent_i == 0:
            cbar = plt.colorbar(sm, label='Agent-0', fraction=0.02, aspect=45, pad=0.01, location='top')
            cbar.ax.tick_params(labelsize=8)
        else:
            cbar = plt.colorbar(sm, label='Agent-1', fraction=0.02, aspect=45, pad=0.01, location='bottom')

def _plot_map_features(map_features):
    for lane in map_features["lane"].values():
        pts = np.array([[p.x, p.y] for p in lane.polyline])
        plt.plot(pts[:, 0], pts[:, 1], linestyle=":", color="gray", linewidth=2)

    for road_line in map_features["road_line"].values():
        pts = np.array([[p.x, p.y] for p in road_line.polyline])
        if road_line.type == 1:
            plt.plot(pts[:, 0], pts[:, 1], 'w', linestyle='dashed', linewidth=2)
        elif road_line.type == 2:
            plt.plot(pts[:, 0], pts[:, 1], 'w', linestyle='solid', linewidth=2)
        elif road_line.type == 3:
            plt.plot(pts[:, 0], pts[:, 1], 'w', linestyle='solid', linewidth=2)
        elif road_line.type == 4:
            plt.plot(pts[:, 0], pts[:, 1], 'xkcd:yellow', linestyle='dashed', linewidth=2)
        elif road_line.type == 5:
            plt.plot(pts[:, 0], pts[:, 1], 'xkcd:yellow', linestyle='dashed', linewidth=2)
        elif road_line.type == 6:
            plt.plot(pts[:, 0], pts[:, 1], 'xkcd:yellow', linestyle='solid', linewidth=2)
        elif road_line.type == 7:
            plt.plot(pts[:, 0], pts[:, 1], 'xkcd:yellow', linestyle='solid', linewidth=2)
        elif road_line.type == 8:
            plt.plot(pts[:, 0], pts[:, 1], 'xkcd:yellow', linestyle='dotted', linewidth=2)
        else:
            plt.plot(pts[:, 0], pts[:, 1], 'k', linewidth=2)

    for road_edge in map_features["road_edge"].values():
        pts = np.array([[p.x, p.y] for p in road_edge.polyline])
        plt.plot(pts[:, 0], pts[:, 1], "k-", linewidth=2)

    for crosswalk in map_features["crosswalk"].values():
        poly_points = [[p.x, p.y] for p in crosswalk.polygon]
        poly_points.append(poly_points[0])
        pts = np.array(poly_points)
        plt.plot(pts[:, 0], pts[:, 1], 'b:', linewidth=2)

    for speed_bump in map_features["speed_bump"].values():
        poly_points = [[p.x, p.y] for p in speed_bump.polygon]
        poly_points.append(poly_points[0])
        pts = np.array(poly_points)
        plt.plot(pts[:, 0], pts[:, 1], 'xkcd:orange', linewidth=2)

    for stop_sign in map_features["stop_sign"].values():
        plt.scatter(stop_sign.position.x, stop_sign.position.y, marker="8", s=100, c="red")


def _plot_traffic_signals(dynamic_map_features):
    for lane_state in dynamic_map_features.lane_states:
        stop_point = lane_state.stop_point

        if lane_state.state in [1, 4, 7]:
            state = 'r' 
        elif lane_state.state in [2, 5, 8]:
            state = 'y'
        elif lane_state.state in [3, 6]:
            state = 'g'
        else:
            state = None

        if state:
            light = plt.Circle((stop_point.x, stop_point.y), 1.2, color=state)
            plt.gca().add_patch(light)
