from pathhack import pkg_path
import time
import numpy as np
import torch
import argparse
import pickle

from src.mif.intention_particle_filter import IntentionParticleFilter
from src.api.pedestrian_intention_application_interface import PedestrianIntentionApplicationInterface
from src.utils import load_preprocessed_train_test_dataset

def arg_parse():
    parser = argparse.ArgumentParser()
    # * Warp LSTM
    # General Options
    parser.add_argument('--dataset_ver', default=0, type=int)
    parser.add_argument('--mode', default='train', help='train, eval, or visual.')
    # Optimization Options
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--clip_grad_norm', default=10., type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    # Model Options
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--end_mask', action='store_true')
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--num_lstms', default=3, type=int)
    parser.add_argument('--embedding_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0., type=float)
    # Evaluation Options
    parser.add_argument('--saved_model_epoch', default=None, type=int)
    parser.add_argument('--visual_batch_size', default=1, type=int)
    parser.add_argument('--visual_num_images', default=10, type=int)
    # Other Options
    parser.add_argument('--save_epochs', default=50, type=int)
    parser.add_argument('--compute_baseline', action='store_true')
    parser.add_argument('--random_seed', default=0, type=int)
    # * Mutable Intention Filter
    # filtering
    parser.add_argument('--num_particles_per_intention', default=10, type=int,\
        help="10, 30, 50, 100.")
    parser.add_argument('--Tf', default=20, type=int, help="12, 20")
    parser.add_argument('--T_warmup', default=2, type=int, help="Number of steps for minimum truncated observation. At least 2 for heuristic in ilm.")
    parser.add_argument('--step_per_update', default=2, type=int, help="1, 2")
    parser.add_argument('--tau', default=10., type=float, help="1., 10.")
    parser.add_argument('--prediction_method', default='wlstm', type=str, help='wlstm or ilm.')
    parser.add_argument('--mutable', action='store_true')
    parser.add_argument('--mutation_prob', default=1e-2, type=float)
    parser.add_argument('--scene', default='edinburgh', type=str)
    # evaluation
    parser.add_argument('--num_top_intentions', default=3, type=int, help='Number of top probability intentions. Options: 1, 3.')
    return parser.parse_args()

# def offline_filtering(
#     trajectory,
#     intention_particle_filter,
#     pedestrian_intention_application_interface):
#     """
#     Inputs:
#         - trajectory: np.ndarray (variable_t, 2).
#     """
#     pass


args = arg_parse()
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

pedestrian_intention_application_interface = PedestrianIntentionApplicationInterface(
    method=args.prediction_method,
    scene=args.scene,
    args=args,
    pkg_path=pkg_path,
    device=device,
)

intention_particle_filter = IntentionParticleFilter(
    pedestrian_intention_application_interface.get_num_intentions(),
    args.num_particles_per_intention,
    pedestrian_intention_application_interface,
)

_, _, _, _, traj_true_test, _ = load_preprocessed_train_test_dataset(pkg_path, dataset_ver=0)
offline_trajectories_dataset = [trajectory.numpy() for trajectory in traj_true_test] # list of np.ndarray (t, 2)
offline_trajectories_indices = np.random.randint(0,len(offline_trajectories_dataset),size=10)
offline_trajectories_set = offline_trajectories_dataset[offline_trajectories_indices] # list of len 10 of np.ndarray (variable_t, 2)

filtering_history_data = []
start_time = time.time()
for trajectory in offline_trajectories_set:
    # trajectory: (variable_t, 2)
    filtering_history = {}
    ground_truth_intention = pedestrian_intention_application_interface.position_to_intention(trajectory[-1])
    intention_particle_filter.reset()
    filtering_history['trajectory'] = trajectory
    filtering_history['ground_truth_intention'] = ground_truth_intention
    filtering_history['last_obs_indices'] = list(range(args.Tf+args.T_warmup, len(trajectory)-args.Tf, args.step_per_update))
    filtering_history['intention_prob_dist'] = []
    filtering_history['fixed_len_predicted_trajectories'] = []
    filtering_history['predicted_trajectories'] = []
    for last_obs_index in filtering_history['last_obs_indices']:
        x_obs = trajectory[:last_obs_index]
        intention_particle_filter.predict(x_obs)
        intention_particle_filter.update_weight(x_obs, tau=args.tau)
        intention_particle_filter.resample()
        if args.mutable:
            intention_particle_filter.mutate(mutation_prob=args.mutation_prob)
        intention_prob_dist = intention_particle_filter.get_intention_probability()
        fixed_len_predicted_trajectories = pedestrian_intention_application_interface.predict_trajectories(
            x_obs,
            intention_particle_filter.get_intention(),
            truncated=True
        ) # (num_particles, Tf, 2)
        predicted_trajectories = pedestrian_intention_application_interface.predict_trajectories(
            x_obs,
            intention_particle_filter.get_intention(),
            truncated=False
        ) # a numpy array of objects. (num_particles, ), where one object is numpy. (variable_t, 2). The predicted trajectories of particles.
        filtering_history['intention_prob_dist'].append(intention_prob_dist)
        filtering_history['fixed_len_predicted_trajectories'].append(fixed_len_predicted_trajectories)
        filtering_history['predicted_trajectories'].append(predicted_trajectories)
    filtering_history_data.append(filtering_history)
print('time to filter: {0:.2f} min'.format((time.time()-start_time)/60.))

result_filename = 'filtering_history_data.p'
with open(result_filename, 'wb') as f:
    pickle.dump(filtering_history_data, f)
    print(result_filename+' is created.')

# logdir = join(pkg_path, 'results', 'mif', args.prediction_method+'_tau_'+str(args.tau))
# result_filename = 'filtering_history.data.p'
# with open(join(logdir, result_filename), 'wb') as f:
#     pickle.dump(infos_list, f)
#     print(result_filename+' is created.')
