from pathhack import pkg_path
import os
import pickle
import numpy as np
from os.path import join
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

class VisualizationMIF:

    def __init__(self, map_info, particle_num_per_intent, infos, sample_true):
        # map_info: contains map information that helps draw the map.
        self.map_info = map_info
        self.particle_num_per_intent = particle_num_per_intent
        (_, _, self.intention_dist_hist, \
             self.particle_intention_hist, self.long_term_pred_hist, self.long_term_true_hist) = infos
        self.intention_dist_hist = np.vstack(self.intention_dist_hist)
        self.sample_true = sample_true
        self.fig = plt.figure(figsize=map_info['fig_size'])
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
        self.axs = []
        for i in range(2):
            self.axs.append(plt.subplot(gs[i]))
        self.num_samples = self.particle_num_per_intent * self.map_info['intent_num']
        plt.tight_layout()
        self.init_plot()

    def init_plot(self):
        # plot intents
        for i in range(self.map_info['intent_num']):
            intent_sq = patches.Rectangle(self.map_info['intent_mean'][i]-self.map_info['intent_radius'], \
                2.*self.map_info['intent_radius'], 2.*self.map_info['intent_radius'], linewidth=1, ls='--', edgecolor=self.map_info['intent_color'][i], facecolor='none')
            self.axs[0].add_patch(intent_sq)
        self.axs[0].set_aspect('equal', 'box')
        self.axs[0].set(xlim=(-1, 18), ylim=(-1, 14))
        self.axs[0].set_yticks([0, 4, 8, 12])
        self.axs[0].set_xticks([0, 4, 8, 12, 16])
        # set up probability distribution plot
        self.axs[1].set_xlim([0, self.map_info['intent_num']+1]) # look symmetric
        self.axs[1].set_ylim([0, 1])
        self.axs[1].set_yticks([0, 0.5, 1])
        self.axs[1].set_xticklabels([])
        self.prob_dist = self.axs[1].bar(range(1, self.map_info['intent_num']+1),\
            1. / self.map_info['intent_num'] * np.ones(self.map_info['intent_num']))
        for j, b in enumerate(self.prob_dist): # k is 0, 1, 2
                b.set_color(self.map_info['intent_color'][j])
        self.prediction_samples = []
        for i in range(self.num_samples):
            particle, = self.axs[0].plot([], [])
            self.prediction_samples.append(particle)
        self.last_obs_point, = self.axs[0].plot([], [], '.', c='k', ms=30)
        self.ped_track, = self.axs[0].plot(self.sample_true[:, 0], self.sample_true[:, 1], 'k-')
        print('Visualization is initialized.')


    def anime_intention_prob_dist(self, interval=20):
        def init():
            pass

        def update(ts):
            sample_full_pred = self.long_term_pred_hist[ts]
            x_true_remained = self.long_term_true_hist[ts]
            particle_intention = self.particle_intention_hist[ts]
            intention_dist = self.intention_dist_hist[ts]
            max_prob_intention = np.argmax(intention_dist)
            for j, b in enumerate(self.prob_dist): # k is 0, 1, 2
                b.set_height(intention_dist[j])
            for i in range(self.num_samples):
                self.prediction_samples[i].set_data([], [])
            for i, (xy_pred, intention) in enumerate(zip(sample_full_pred, particle_intention)):
                if intention == max_prob_intention:
                    xy_pred = xy_pred - xy_pred[0] + x_true_remained[0] # refine the start
                    self.prediction_samples[i].set_data(xy_pred[:,0], xy_pred[:,1])
                    self.prediction_samples[i].set_color(self.map_info['intent_color'][intention])
            self.last_obs_point.set_data(x_true_remained[0, 0], x_true_remained[0, 1])

        ani = FuncAnimation(self.fig, update, frames=self.intention_dist_hist.shape[0],
                            init_func=init, interval=interval, repeat=True)
        plt.show()


def intention_hist_plot(intention_dist_hist):
    hues = [[215, 25, 28], [253, 174, 97], [44, 123, 182]]
    for i in range(len(hues)):
        hues[i] = np.array(hues[i])/255.
    intention_dist_hist = np.stack(intention_dist_hist)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.plot(intention_dist_hist[:, 0], c=hues[0], lw=3, label='intention 0')
    ax.plot(intention_dist_hist[:, 1], c=hues[1], lw=3, label='intention 1')
    ax.plot(intention_dist_hist[:, 2], c=hues[2], lw=3, label='intention 2')
    ax.axvline(x=30, c='k', ls='--')
    ax.axvline(x=86-12-2, c='k', ls='--')
    ax.axvline(x=150, c='k', ls='--')
    intent_gt_1 = patches.Rectangle((0, 1.05), 86-12-2, 1.1, facecolor=hues[1])
    ax.add_patch(intent_gt_1)
    intent_gt_2 = patches.Rectangle((86-12-2, 1.05), len(intention_dist_hist)-(86-12-2)-0.5, 1.1, facecolor=hues[0])
    ax.add_patch(intent_gt_2)
    ax.set(xlim=(-5, 205), ylim=(-0.1, 1.1))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    fig_filename = 'intention_estimation_history.pdf'
    fig_filepath = join(pkg_path, 'results/visual', fig_filename)
    plt.savefig(fig_filepath)
    print(fig_filepath+' is created.')


def filtering_prediction_plot(infos):
    hues = [[215, 25, 28], [253, 174, 97], [44, 123, 182]]
    for i in range(len(hues)):
        hues[i] = np.array(hues[i])/255.
    dataset_filename = 'original_traj_intention_changing_animation.p'
    dataset_filepath = join(pkg_path, 'results/visual', dataset_filename)
    with open(dataset_filepath, 'rb') as f:
        abnormal_traj_data = pickle.load(f)
        print(dataset_filepath+' is loaded.')
    num_abnormal_samples = 0
    for data_i in abnormal_traj_data:
        num_abnormal_samples = num_abnormal_samples + len(data_i)
    print('Number of abnormal samples: ', num_abnormal_samples)
    abnormal_traj_data_total = abnormal_traj_data[0]+abnormal_traj_data[1]+abnormal_traj_data[2]
    sample_true_abnormal = abnormal_traj_data_total[0]
    (correct_intention, percentage_hist, intention_dist_hist, \
            particle_intention_hist, long_term_pred_hist, long_term_true_hist) = infos
    for time_idx in [30, 150]:
        sample_full_pred = long_term_pred_hist[time_idx]
        x_true_remained = long_term_true_hist[time_idx]
        particle_intention = particle_intention_hist[time_idx]
        intention_dist = intention_dist_hist[time_idx]
        max_prob_intention = np.argmax(intention_dist)
        for xy_pred, intention in zip(sample_full_pred, particle_intention):
            if intention == max_prob_intention:
                xy_pred = xy_pred - xy_pred[0] + x_true_remained[0] # refine the start
                plt.plot(xy_pred[:,0], xy_pred[:,1], c=hues[intention])
        plt.plot(x_true_remained[0, 0], x_true_remained[0, 1], '.', c='k', ms=15, label='last observation') # int
    plt.plot(sample_true_abnormal[:, 0], sample_true_abnormal[:, 1],'k-')
    time_intention_changing_idx = 86-12-2
    x_true_remained = long_term_true_hist[time_intention_changing_idx]
    plt.plot(x_true_remained[0, 0], x_true_remained[0, 1], '.', c='k', ms=15, label='last observation') # int
    plt.xlim(-1, 18)
    plt.ylim(-1, 14)
    plt.yticks([0, 4, 8, 12])
    plt.xticks([0, 4, 8, 12, 16])
    plt.tight_layout()
    fig_filename = 'trajectory_prediction_snapshots_with_top_probability_intention.pdf'
    fig_filepath = join(pkg_path, 'results/visual', fig_filename)
    plt.savefig(fig_filepath)
    print(fig_filepath+' is created.')


def main_animation():
    r"""Animate the filtering process."""
    # initilize visualization settings
    map_info = {}
    map_info['intent_num'] = 3
    map_info['intent_mean'] = np.array([[ 7.6491633 , 11.74338086],
                                        [ 3.00575615,  0.77987421],
                                        [15.72789116,  7.75681342],])
    map_info['intent_color'] = [[215, 25, 28], [253, 174, 97], [44, 123, 182]]
    for i in range(len(map_info['intent_color'])):
        map_info['intent_color'][i] = np.array(map_info['intent_color'][i])/255.
    map_info['intent_radius'] = 0.8
    map_info['fig_size'] = (12, 7)
    particle_num_per_intent = 200
    # load filtering data
    filtering_results_filename = 'infos_intention_changing_animation.p'
    filtering_results_filepath = join(pkg_path, 'results/visual', filtering_results_filename)
    with open(filtering_results_filepath, 'rb') as f:
        infos_test = pickle.load(f)
        print(filtering_results_filepath + ' is loaded.')
    dataset_filename = 'original_traj_intention_changing_animation.p'
    dataset_filepath = join(pkg_path, 'results/visual', dataset_filename)
    with open(dataset_filepath, 'rb') as f:
        abnormal_traj_data = pickle.load(f)
        print(dataset_filepath+' is loaded.')
    sample_true_abnormal = abnormal_traj_data[0][0]
    vis_mif = VisualizationMIF(map_info, particle_num_per_intent, infos_test['200_12_True'], sample_true_abnormal)
    vis_mif.anime_intention_prob_dist()


def main_plot():
    r"""Create snapshots of trajectory prediction with top probability intention and intention estimation history."""
    filtering_results_filename = 'infos_intention_changing_animation.p'
    filtering_results_filepath = join(pkg_path, 'results/visual', filtering_results_filename)
    with open(filtering_results_filepath, 'rb') as f:
        infos_test = pickle.load(f)
        print(filtering_results_filepath + ' is loaded.')
    filtering_prediction_plot(infos_test['200_12_False'])
    (correct_intention, percentage_hist, intention_dist_hist, \
        particle_intention_hist, long_term_pred_hist, long_term_true_hist) = infos_test['200_12_False']
    intention_dist_hist = np.vstack(intention_dist_hist)
    intention_hist_plot(intention_dist_hist)


if __name__ == '__main__':
    main_animation() # animate the filtering process
    # main_plot() # create snapshots of trajectory prediction with top probability intention and intention estimation history






