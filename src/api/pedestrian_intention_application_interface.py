import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.api.intention_application_interface import IntentionApplicationInterface
from src.utils import load_warplstm_model_by_arguments

"""
comments for pedestrian application
    - intention_mean
    - num_intentions
        the number of intentions.
    - num_particles_per_intention
        the number of particles set for each intention.
    - num_tpp (default: 12)
        number of trajectory points for prediction.

    # self.intention_sampler = intention_sampler
    # self.intention_coordinates = self.intention_sampler.intention_bottomleft_coordinates + self.intention_sampler.intention_width
    # self.num_tpp = num_tpp
    # intention_sampler,
    # num_tpp=12,
    # if self.args.intention_application == 'pedestrian': 
    #     self.intention_application_interface = PedestrianIntentionApplicationInterface(args)


# oe = self.x_pred - x_pred_true[np.newaxis]# offset error (particle_num, num_tpp, 2)
# aoe = np.mean(np.linalg.norm(oe, axis=2), axis=1) # (particle_num,) # ! gap is aoe
# self.weight *= np.exp(-tau*aoe)
# self.weight /= self.weight.sum()


    # def predict(self, x_obs, pred_func):
    #     self.goals = self.intention2goal()
    #     self.x_pred, _ = pred_func(x_obs, self.goals, self.intention, self.intention_coordinates, num_intentions=self.num_intentions, num_tpp=self.num_tpp)
    #     return self.x_pred
    
    # def predict_till_end(self, x_obs, long_pred_func):
    #     self.goals = self.intention2goal()
    #     _, infos = long_pred_func(x_obs, self.goals, self.intention, self.intention_coordinates,  num_intentions=self.num_intentions, num_tpp=self.num_tpp)
    #     return infos

    def particle_weight_intention_prob_dist(self):
        ### soft weight ###
        # weight_balanced = np.log(self.weight+params.WEIGHT_EPS) - np.log(min(self.weight+params.WEIGHT_EPS)) + 1. # (no zero in it is convenient for sampling)
        # weight_balanced /= sum(weight_balanced)
        ### original weight ###
        weight_balanced = self.weight
        particle_weight = [] # list of num_intentions (3) lists
        particle_weight_with_zero = self.weight * self.intention_mask# (particle_num,) * (num_intentions, particle_num) -> (num_intentions, particle_num)
        for intention_index in range(self.num_intentions):
            particle_weight_intention = particle_weight_with_zero[intention_index, particle_weight_with_zero[intention_index, :].nonzero()]
            particle_weight.append(np.squeeze(particle_weight_intention, axis=0))
        return particle_weight, (weight_balanced*self.intention_mask).sum(axis=1)
    
    def intention2goal(self):
        return self.intention_sampler.idx2intent_sampling(self.intention)
"""

class PedestrianIntentionApplicationInterface(IntentionApplicationInterface):
    def __init__(
        self,
        method,
        scene,
        args,
        pkg_path,
        device,
    ):
        """
        A child class of intention application interface for pedestrian applications. 

        Initialize with name of the application.

        Inputs:
            - method: Trajectory prediction method. 'ilm' for intention-aware linear model, 
            or 'wlstm' for warp lstm.
            - scene: e.g. 'edinburgh'.
            - args: arguments related to method.
                - Tf: lookahead time window.
            - pkg_path: absolute path of the package.
            - device: 'cuda:0' or 'cpu'.

        Updated:
            - self.application: 'pedestrian2D'.
            - self.method
            - self.scene
            - self.device
            - self.Tf: lookahead time window.
            - self.model
            - self.pedestrian_intention_sampler
        
        Outputs:
            - None
        """
        super(PedestrianIntentionApplicationInterface, self).__init__('pedestrian2D')
        self.method = method
        self.scene = scene
        self.device = device
        self.Tf = args.Tf
        # load model of the method
        if self.method == 'wlstm':
            # a list of models with different observation ratios
            self.model = []
            for dataset_ver in [0, 25, 50, 75]:
                args.dataset_ver = dataset_ver
                self.model.append(load_warplstm_model_by_arguments(args, pkg_path, self.device))
        elif self.method == 'ilm':
            self.model = None
        else:
            raise RuntimeError('Wrong method input for PedestrianIntentionApplicationInterface.')
        # load sampler of the scene
        if self.scene == 'edinburgh':
            self.pedestrian_intention_sampler = load_edinburgh_pedestrian_intention_sampler()
        else:
            raise RuntimeError('Scene is not found.')
        return
    
    def propagate_x(self, x_est, intention, intention_mask, x_obs=None):
        """
        In pedestrian application, state estimate is not propagated.
        x_est will be replaced instead of propagated in the next iteration of filtering.
        The input x_est could be None (initialized) or numpy :math:`(num_particles, T_f, 2)`, 
        but they are not used in propagate_x() in pedestrian application.

        x_est is the trajectory prediction during the period :math:`[t-T_f+1, t]`, where 
        :math:`t` is the current time step, and :math:`T_f` is the truncated time window.
        x_obs is the observed trajectory during the period :math:`[1, t]`. Only the observation 
        during the period :math:`[1, t-T_f]` will be used to make prediction in the period 
        :math:`[t-T_f+1, t]`.

        Inputs:
            - x_est: numpy. :math:`(num_particles, T_f, 2)` or None. The predicted trajectories of particles. 
            - intention: numpy. :math:`(num_particles,)` Intention hypotheses for all particles. 
            e.g. for num_intentions=3, num_particles_per_intention=5, intention = 
            array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).
            - intention_mask: numpy. :math:`(num_intentions, num_particles)` Mask on intention 
            hypotheses of all particles. # ! may not be required.
            e.g. for intention = array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            intention_mask = \n
            array([[ True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False],\n
                   [False, False, False, False, False,  True,  True,  True,  True,  True, False, False, False, False, False],\n
                   [False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True]])
            - x_obs: numpy. :math:`(t, 2)`. The trajectory observed from beginning till the current 
            time step, i.e., :math:`[1,t]`. Only :math:`[1,t-T_f]` will be used.
        
        Updated:
            - None
        
        Outputs:
            - x_est: numpy. :math:`(num_particles, T_f, 2)`. The prediction during the period 
            :math:`[t-T_f+1, t]`.
        """
        x_obs_truncated = x_obs[:-self.Tf] # (t-Tf, 2)
        x_est = None
        return x_est

        # for last_obs_index in range(num_tpp+warmup_step, len(sample_true)-num_tpp, step_per_update): # not include 0.
        # x_pred_true = sample_true[last_obs_index-num_tpp+1:last_obs_index+1].numpy()
        # x_obs = sample_true[:last_obs_index-num_tpp+1].numpy()
        # x_true_remained = sample_true[last_obs_index+1:].numpy()
        # percentage_curr = float(last_obs_index/len(sample_true))

    def predict_intention_aware_linear_model(self, x_obs, intention, truncated=False):
        """
        Predict trajectories given observation and intention hypotheses.

        Inputs:
            - x_obs: numpy. The observed trajectory. Can be either :math:`(t-T_f, 2)` or 
            math:`(t, 2)`, depending on whether we do propagate_x() or long term prediction.
            - intention: numpy. :math:`(num_particles,)` Intention hypotheses for all particles. 
            e.g. for num_intentions=3, num_particles_per_intention=5, intention = 
            array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).
            - truncated: True means we use prediction function in propagate_x(). False means we are 
            performing long term prediction until the goal position.
            
        Updated:
            - None
        
        Outputs:
            - x_est: If truncated=True, numpy. :math:`(num_particles, T_f, 2)`. 
            Otherwise a numpy array of objects. (num_particles, ), where one object is numpy. :math:`(variable_t, 2)`.
        """
        # num_intentions = self.pedestrian_intention_sampler.num_intentions
        num_particles = len(intention)
        if truncated:
            x_est = np.empty((num_particles, self.Tf, 2))
        else:
            x_est = np.empty(num_particles).astype(object)
        heuristic_distance = np.linalg.norm(self.pedestrian_intention_sampler.intention_center_coordinates \
            - x_obs[-1], axis=1) # np (num_intentions,)
        mean_vel_mag = np.mean(np.linalg.norm(x_obs[1:]-x_obs[:-1], axis=1))
        step_num_noise = np.random.randint(2, 10, size=self.pedestrian_intention_sampler.num_intentions)
        heuristic_num_steps = (heuristic_distance/mean_vel_mag).astype(int)+step_num_noise
        heuristic_num_steps[heuristic_num_steps<self.Tf+1] = self.Tf+1
        
        goal_position_samples = self.pedestrian_intention_sampler.sampling_goal_positions(intention)
        survived_intention_indices = np.unique(intention)
        for i in survived_intention_indices:
            goal_position_samples_i = goal_position_samples[intention==i]
            # ! Zhe pause here


        for i in range(intention_num):
            sample_goals_i = sample_goals[intentions==i]
            sample_base_pred_i = (np.linspace(x_obs[-1], sample_goals_i, num=heuristic_num_steps[i])).transpose(1,0,2) # (batch, time_step, 2)
            x_obs_copy = x_obs[np.newaxis, :-1] * np.ones((sample_base_pred_i.shape[0],1,1))
            sample_base_i = np.concatenate((x_obs_copy, sample_base_pred_i), axis=1)
            sb = sample_base_i
            sample_full_pred_arr[np.where(intentions==i)[0]]=list(sb[:, -heuristic_num_steps[i]:])
            sample_pred[intentions==i] = sb[:, -heuristic_num_steps[i]:-heuristic_num_steps[i]+num_tpp]
        return sample_pred, sample_full_pred_arr

        pass
    

    def ilm_pred_fit(self, x_obs, sample_goals, intentions, intention_coordinates, \
        intention_num=3, num_tpp=12, silent=True):
        """
        inputs:
            - x_obs: np. (t, 2)

        outputs:
            - sample_pred: np. (340, 20, 2)
            - sample_full_pred_arr: np array of objects. (340, )
                - sample: np. (variable_t, 2)
        """
        sample_full_pred_arr = np.empty(len(intentions)).astype(object)
        sample_pred = np.empty((len(intentions), num_tpp, 2))
        heuristic_dist = np.linalg.norm(intention_coordinates-x_obs[-1], axis=1)
        mean_vel_mag = np.mean(np.linalg.norm(x_obs[1:]-x_obs[:-1], axis=1))
        step_num_noise = np.random.randint(2, 10, size=intention_num)
        heuristic_num_steps = (heuristic_dist/mean_vel_mag).astype(int)+step_num_noise
        heuristic_num_steps[heuristic_num_steps<num_tpp+1] = num_tpp+1
        for i in range(intention_num):
            sample_goals_i = sample_goals[intentions==i]
            sample_base_pred_i = (np.linspace(x_obs[-1], sample_goals_i, num=heuristic_num_steps[i])).transpose(1,0,2) # (batch, time_step, 2)
            x_obs_copy = x_obs[np.newaxis, :-1] * np.ones((sample_base_pred_i.shape[0],1,1))
            sample_base_i = np.concatenate((x_obs_copy, sample_base_pred_i), axis=1)
            sb = sample_base_i
            sample_full_pred_arr[np.where(intentions==i)[0]]=list(sb[:, -heuristic_num_steps[i]:])
            sample_pred[intentions==i] = sb[:, -heuristic_num_steps[i]:-heuristic_num_steps[i]+num_tpp]
        return sample_pred, sample_full_pred_arr

     


def load_edinburgh_pedestrian_intention_sampler():
    """
    Create a intention sampler for 'edinburgh' scene. Unit: meter.

    Inputs:
        - None

    Outputs:
        - edinburgh_pedestrian_intention_sampler: an instance of PedestrianIntentionSampler.
    """
    intention_width = 1.5
    # bottom left coordinates of the potential intentions 
    x_bottom_left_list = []
    x_bottom_left = np.array([2.7, 0.2])
    x_bottom_left_list.append(x_bottom_left)
    for i in np.arange(1.5, 1.5+1.5*8, 1.5):
        x_bottom_left_list.append(x_bottom_left+np.array([i, 0]))
    x_bottom_left = x_bottom_left_list[-1]
    for j in np.arange(1.5, 1.5+1.5*7, 1.5):
        x_bottom_left_list.append(x_bottom_left+np.array([0, j]))
    x_bottom_left = x_bottom_left_list[-1]
    for i in np.arange(1.5, 1.5+1.5*10, 1.5):
        x_bottom_left_list.append(x_bottom_left+np.array([-i, 0]))
    x_bottom_left = x_bottom_left_list[-1]
    for j in np.arange(1.5, 1.5+1.5*4, 1.5):
        x_bottom_left_list.append(x_bottom_left+np.array([0, -j]))
    x_bottom_left = x_bottom_left_list[-1]
    x_bottom_left_list.append(x_bottom_left+np.array([1.5, 0.]))
    x_bottom_left = x_bottom_left_list[-1]
    x_bottom_left_list.append(x_bottom_left+np.array([0., -1.5]))
    x_bottom_left = x_bottom_left_list[-1]
    x_bottom_left_list.append(x_bottom_left+np.array([1.5, 0.]))
    x_bottom_left = x_bottom_left_list[-1]
    x_bottom_left_list.append(x_bottom_left+np.array([0., -1.5]))
    edinburgh_intention_bottomleft_coordinates = np.stack(x_bottom_left_list, axis=0)
    num_intentions = len(edinburgh_intention_bottomleft_coordinates)
    edinburgh_scene_xlim, edinburgh_scene_ylim = (-1, 16), (-1, 14)
    edinburgh_pedestrian_intention_sampler = PedestrianIntentionSampler(
        intention_width,
        num_intentions,
        edinburgh_intention_bottomleft_coordinates,
        edinburgh_scene_xlim,
        edinburgh_scene_ylim,
    )
    return edinburgh_pedestrian_intention_sampler




class PedestrianIntentionSampler:
    def __init__(
        self,
        intention_width,
        num_intentions,
        intention_bottomleft_coordinates,
        scene_xlim,
        scene_ylim,
    ):
        """
        Pedestrian intention is defined as a 2D square-shaped goal region. This sampler includes intention 
        information, and can sample pedestrian goal positions given the intention hypotheses.

        Initialize with the intention information.

        Inputs:
            - intention_width: Width of the square goal region.
            - num_intentions: Number of all potential intentions.
            - intention_bottomleft_coordinates: numpy. :math:`(num_intentions, 2)` Bottom left coordinates 
            of the squared intention regions.
            - scene_xlim: tuple. :math:`(2,)` x boundary of the scene.
            - scene_ylim: tuple. :math:`(2,)` y boundary of the scene.

        Updated:
            - self.intention_width
            - self.num_intentions
            - self.intention_bottomleft_coordinates: numpy. :math:`(num_intentions, 2)`
            - self.intention_center_coordinates: numpy. :math:`(num_intentions, 2)`
            - self.scene_xlim: tuple. :math:`(2,)`
            - self.scene_ylim: tuple. :math:`(2,)`
        
        Outputs:
            - None
        """
        self.intention_width = intention_width
        self.num_intentions = num_intentions
        self.intention_bottomleft_coordinates = intention_bottomleft_coordinates
        self.intention_center_coordinates = self.intention_bottomleft_coordinates \
            + self.intention_width/2.
        self.scene_xlim, self.scene_ylim = scene_xlim, scene_ylim
        return
        
    def position_to_intention(self, position):
        """
        Identify to which the intention the input position belongs.

        Inputs:
            - position: numpy. :math:`(2,)`

        Updated:
            - None
        
        Outputs:
            - intention_index: integer, or None if the position does not belong to any intention. 
        """
        # ! Original function name: belong2intent
        vec_bl_to_pos = position - self.intention_bottomleft_coordinates
        bool_flag = ((vec_bl_to_pos[:, 0] > 0) * (vec_bl_to_pos[:, 0] < self.intention_width)) \
            * ((vec_bl_to_pos[:, 1] > 0) * (vec_bl_to_pos[:, 1] < self.intention_width))
        if len(np.where(bool_flag)[0]) == 0:
            return None
        else:
            return np.where(bool_flag)[0][0] # index of intention
    
    def sampling_goal_positions(self, intention_indices):
        """
        Sample goal positions given intention indices. One goal position is sampled for
        each intention index.

        Inputs:
            - intention_indices: numpy. :math:`(num_particles,)` indices of intention 
            hypotheses from particles.

        Updated:
            - None
        
        Outputs:
            - goal_position_samples: numpy. :math:`(num_particles, 2)` samples of goal 
            positions.
        """
        # ! Original function name: idx2intent_sampling
        intention_bl_samples = self.intention_bottomleft_coordinates[intention_indices]
        samples_noise = np.random.uniform(low=0., high=self.intention_width, \
            size=intention_bl_samples.shape)
        goal_position_samples = intention_bl_samples + samples_noise
        return goal_position_samples

    def uniform_sampling_goal_positions(self, num_samples_per_intention=20):
        """
        Sample a fixed amount of goal positions from each potential intention.

        Inputs:
            - num_samples_per_intention: The number of goal position samples for each 
            intention.

        Updated:
            - None
        
        Outputs:
            - goal_position_samples: numpy. 
            :math:`(num_intentions*num_samples_per_intention, 2)` 
            The goal position samples from all potential intentions.
        """
        # ! Original function name: uniform_sampling
        samples_noise = np.random.uniform(low=0., high=self.intention_width, \
            size=(self.num_intentions, num_samples_per_intention, 2))
        goal_position_samples = samples_noise \
            + self.intention_bottomleft_coordinates[:, np.newaxis, :]
        goal_position_samples = goal_position_samples.reshape(-1, 2)
        return goal_position_samples
    
    def visualize_intention(self, goal_positions=None):
        """
        Visualize the scene with intentions and end positions of trajectories if any.

        Inputs:
            - goal_positions: None or numpy. :math:`(num_trajectories, 2)` End positions of trajectories.

        Updated:
            - None
        
        Outputs:
            - fig
            - ax
        """
        fig, ax = plt.subplots()
        if goal_positions is not None:
            ax.plot(goal_positions[:, 0], goal_positions[:, 1], 'b.')
        for intent_bottom_left in self.intention_bottomleft_coordinates:
            rect = patches.Rectangle(intent_bottom_left, self.intention_width, \
                self.intention_width,linewidth=1,edgecolor='k',facecolor='none',zorder=3)
            ax.add_patch(rect)
        ax.set(xlim=self.scene_xlim, ylim=self.scene_ylim)
        ax.axis('scaled')
        return fig, ax