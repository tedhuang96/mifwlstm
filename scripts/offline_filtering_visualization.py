from pathhack import pkg_path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


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

result_filename = 'filtering_history_data_wlstm.p'
with open(result_filename, 'rb') as f:
    filtering_history_data = pickle.load(f)

print(len(filtering_history_data)) # 10 trajectories
print(filtering_history_data[0].keys())

filtering_history = filtering_history_data[5]

trajectory = filtering_history['trajectory']

fig, ax = pedestrian_intention_application_interface.pedestrian_intention_sampler.visualize_intention()
print(len(ax.patches))
print(len(filtering_history['intention_prob_dist'][0]))
trajectory_plot, = ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-')
predictions_plot = []
last_observed_position, = ax.plot([], [], '.', c='k', ms=20)
for i in range(intention_particle_filter.num_particles):
    prediction_plot, = ax.plot([],[],'r-')
    predictions_plot.append(prediction_plot)

def init():
    pass

def animate(ts):
    plot_alpha = 0.7
    hues = [[215,48,39], [244,109,67], [253,174,97], [171,217,233], [116,173,209], [69,117,180]]
    for i in range(len(hues)):
        hues[i] = np.array(hues[i])/255.
    last_obs_index = filtering_history['last_obs_indices'][ts]
    predicted_trajectories = filtering_history['fixed_len_predicted_trajectories'][ts]
    intention_prob_dist = filtering_history['intention_prob_dist'][ts]
    x_obs = trajectory[:last_obs_index]
    predicted_trajectories = predicted_trajectories - predicted_trajectories[:,:1,:] + x_obs[-1] # account for trajectory offset
    for prediction_plot, predicted_trajectory in zip(predictions_plot, predicted_trajectories):
        prediction_plot.set_data(predicted_trajectory[:,0], predicted_trajectory[:,1])
    last_observed_position.set_data(x_obs[-1,0], x_obs[-1,1])
    for patch in ax.patches:
        patch.set_facecolor('w')
    top_intentions = np.flip(np.argsort(intention_prob_dist))[:3] # ndarray
    ax.patches[top_intentions[0]].set_facecolor(list(hues[0])+[plot_alpha])
    ax.patches[top_intentions[1]].set_facecolor(list(hues[2])+[plot_alpha])
    ax.patches[top_intentions[2]].set_facecolor(list(hues[4])+[plot_alpha])

ani = FuncAnimation(fig, animate, frames=len(filtering_history['last_obs_indices']),
                            init_func=init, interval=20, repeat=True)

plt.show()