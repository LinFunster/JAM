import sys
sys.path.append("..")
import torch
import argparse
import torch.functional as F
from torch.utils.data import DataLoader
import sys
import tarfile
import argparse
import os
import torch
from utils.inter_pred_utils import *
from collections import OrderedDict

from waymo_open_dataset.protos.motion_submission_pb2 import *

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

def transform_to_global_frame_multimodal_submission(trajectories, ego_pose):
    # trajectories shape: [2, 6, 80, 2]
    ego_p = ego_pose[:2]  # Extract the ego position (x, y)
    ego_h = ego_pose[3]   # Extract the ego heading angle (heading)
    agent_N = trajectories.shape[0]  # 2
    modal_N = trajectories.shape[1]  # 6
    global_trajectories = []
    for agent_i in range(agent_N):
        multimodal_trajs = []
        for modal_i in range(modal_N):
            # Convert trajectory points into LineString objects.
            line = LineString(trajectories[agent_i, modal_i])
            # Rotate the trajectory to align with the global coordinate system.
            line = rotate(line, ego_h, origin=(0, 0), use_radians=True)
            # Translate the trajectory to the global coordinate system.
            line = affine_transform(line, [1, 0, 0, 1, ego_p[0], ego_p[1]])
            # Convert the LineString back to a NumPy array.
            traj = np.array(line.coords)
            multimodal_trajs.append(traj)
        global_trajectories.append(multimodal_trajs)

    return np.array(global_trajectories)

def save_to_file(scnerio_predictions, submission_info):
    print('saving....')
    submission_type=2  # 2 interactive else 1

    submission = MotionChallengeSubmission(
        account_name=submission_info['account_name'], 
        unique_method_name=submission_info['unique_method_name'],
        authors=submission_info['authors'], 
        affiliation=submission_info['affiliation'], 
        description=submission_info['description'],
        submission_type=submission_type, 
        scenario_predictions=scnerio_predictions,
        uses_lidar_data=submission_info['uses_lidar_data'],
        uses_camera_data=submission_info['uses_camera_data'],
        uses_public_model_pretraining=submission_info['uses_public_model_pretraining'],
        public_model_names=submission_info['public_model_names'],
        num_model_parameters=submission_info['num_model_parameters']
        )

    base_path = f"{args.sub_output_dir}/{args.name}/"
    save_path = base_path + f"{args.name}.proto"
    tar_path = base_path + f"{args.name}.gz"
    f = open(save_path, "wb")
    f.write(submission.SerializeToString())
    f.close()

    print(f'Testing_saved:{tar_path},zipping...')

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(save_path)
        tar.close()

    print('Finished!')

def wrap2submission(pred_trajs, pred_scores, object_id, scenario_id, scenario_predictions, current_xyzh):
    pred_trajs=pred_trajs.cpu().numpy()
    pred_scores=pred_scores.cpu().numpy()
    current_xyzh=current_xyzh.numpy()
    B = len(pred_trajs)
    for b in range(B):
        # transform pred_trajs to global frame
        pred_trajs_global = transform_to_global_frame_multimodal_submission(pred_trajs[b], current_xyzh[b])
        scenario_preds = wrap2submission_one(pred_trajs_global, pred_scores[b], object_id[b], scenario_id[b])
        scenario_predictions.append(scenario_preds)
    return scenario_predictions

def wrap2submission_one(pred_trajs, pred_scores, object_id, scenario_id):
    assert len(pred_trajs)==2
    full_scored_trajs = []
    for j in range(6):
        object_trajs = []
        for i in range(2):
            center_x = pred_trajs[i, j, :, 0]
            center_y = pred_trajs[i, j, :, 1]
            traj = Trajectory(center_x=center_x, center_y=center_y)
            score_traj = ObjectTrajectory(object_id=object_id[i], trajectory=traj)
            object_trajs.append(score_traj)   
        full_scored_trajs.append(ScoredJointTrajectory(trajectories=object_trajs, confidence=pred_scores[j]))
    joint_prediction = JointPrediction(joint_trajectories=full_scored_trajs)
    return ChallengeScenarioPredictions(scenario_id=scenario_id, joint_prediction=joint_prediction)


def test(model, test_data):
    scenario_predictions = []
    size = len(test_data)*args.batch_size
    with torch.no_grad():
        for i, batch_dict in enumerate(test_data):
            # prepare data
            inputs = {
                'ego_state': batch_dict[0].to(args.device),
                'neighbors_state': batch_dict[1].to(args.device),
                'map_lanes': batch_dict[2].to(args.device),
                'map_crosswalks': batch_dict[3].to(args.device),
            }
            object_id = batch_dict[4]
            scene_id = batch_dict[5]
            current_xyzh = batch_dict[6]

            # query the model
            outputs = model(inputs)

            # compute metrics (JAM)
            outputs = model(inputs)
            pred_trajs = outputs[f'joint_interactions'][:, :, :, 4::5, :2] # [B,N,M,16,2]
            pred_scores = outputs[f'joint_scores'] # [B,N,M]
            pred_scores = pred_scores.sum(1)
            pred_scores = F.softmax(pred_scores,dim=-1) # [B,M]

            scenario_predictions = wrap2submission(pred_trajs, pred_scores, object_id, scene_id, scenario_predictions, current_xyzh)
            sys.stdout.write(f'\rProcessed:{i*args.batch_size}-{size}')
            sys.stdout.flush()

    return scenario_predictions    

def main():
    log_path = f"{args.sub_output_dir}/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')
    logging.info("------------- {} -------------".format(args.name))
    logging.info("save name: {}".format(args.name))
    logging.info("seed: {}".format(args.seed))
    logging.info("test_set: {}".format(args.test_set))
    set_seed(args.seed)

    submission_info = dict(
        account_name='xxx@email.com',
        unique_method_name='method name',
        authors=['authors name'], 
        description="None", 
        affiliation='None',
        uses_lidar_data=False,
        uses_camera_data=False,
        uses_public_model_pretraining=False,
        public_model_names='N/A',
        num_model_parameters='N/A',
    )

    from model.JAM import JAM
    jam_model = JAM(modalities=6,class_query_topK=1,encoder_layers=6,future_len=80,neighbors_to_predict=1,class_query_N=64).to(args.device)
    # load param
    model_ckpts = load_model_from_ddp_ckpts(args.jam_model_path, args.device)
    jam_model.load_state_dict(model_ckpts)
    jam_model.eval()
    total_params = sum(p.numel() for p in jam_model.parameters())
    logging.info(f"jam total parameters: {total_params}")
    test_dataset = DrivingData(args.test_set+'/*.npz', test_set=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)
    scenario_predictions = test(jam_model, test_loader)
    save_to_file(scenario_predictions, submission_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--name', type=str, help='log save path (default: "JAM")', default="test_JAM") 
    parser.add_argument('--sub_output_dir', type=str, help='path to save output submission file', default="/JAM/submission_pkg")
    parser.add_argument('--test_set', type=str, help='path to validation data', default="/JAM/waymo_dataset_1_2/submission_testset") 
    parser.add_argument('--jam_model_path', type=str, help='path to saved model', default="/JAM/jam_log/jam/epochs_29.pth")

    parser.add_argument('--seed', type=int, help='fix random seed', default=3407) 
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--device', type=str, help='run on which device', default='cuda')
    args = parser.parse_args()
    main()
