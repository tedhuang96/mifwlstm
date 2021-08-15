import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.api.intention_application_interface import IntentionApplicationInterface
from src.utils import load_warplstm_model_by_arguments


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
    
    def propagate_x(self, x_est, intention, x_obs=None):
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
            - x_est: None. The predicted trajectories of particles. It is not passed from the 
            previous generation of particles. So in the input x_est should be None.
            - intention: numpy. :math:`(num_particles,)` Intention hypotheses for all particles. 
            e.g. for num_intentions=3, num_particles_per_intention=5, intention = 
            array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).
            - x_obs: numpy. :math:`(t, 2)`. The trajectory observed from beginning till the current 
            time step, i.e., :math:`[1,t]`. Only :math:`[1,t-T_f]` will be used.
        
        Updated:
            - None
        
        Outputs:
            - x_est: numpy. :math:`(num_particles, T_f, 2)`. The prediction during the period 
            :math:`[t-T_f+1, t]`.
        """
        x_obs_truncated = x_obs[:-self.Tf] # (t-Tf, 2)
        x_est = self.predict_trajectories(x_obs_truncated, intention, truncated=True)
        return x_est
    
    def compare_observation(self, x_est, x_obs):
        """
        Compute the difference between the state estimate against observation. 

        Inputs:
            - x_est: numpy. :math:`(num_particles, T_f, 2)`. The prediction during the period 
            :math:`[t-T_f+1, t]`.
            - x_obs: numpy. :math:`(t, 2)`. The trajectory observed from beginning till the current 
            time step, i.e., :math:`[1,t]`. Only :math:`[t-T_f+1, t]` will be used.

        Updated:
            - None
        
        Outputs:
            - gap: numpy. :math:`(num_particles,)`. The difference between x_est and x_obs.
        """
        if x_est is None:
            raise RuntimeError('x_est is None during comparison against x_obs.')
        elif isinstance(x_est, np.ndarray):
            x_obs_compared = x_obs[np.newaxis, -self.Tf:] # (1, Tf, 2)
            compared_offset_error = x_est - x_obs_compared # (num_particles, T_f, 2)
            gap = np.mean(np.linalg.norm(compared_offset_error, axis=2), axis=1) # (num_particles,)
            return gap
        else:
            raise RuntimeError('x_est is not numpy.ndarray for compare_observation() in '+\
                'PedestrianIntentionApplicationInterface.')

    def resample_x(self, x_est, resampled_indices):
        """
        Use the resampled indices to re-organize the state estimates.

        Inputs:
            - x_est: list of length :math:`num_particles` or None. The state estimates of particles. 
            - resampled_indices: numpy. :math:`(num_particles,)` The resampled integer indices of 
            particles.

        Updated:
            - None
        
        Outputs:
            - resampled_x_est: None. In pedestrian application, x_est is not passed to the 
            resampled particles.
        """
        if x_est is None:
            return None
        elif isinstance(x_est, np.ndarray):
            return None
        else:
            raise RuntimeError('x_est is not numpy.ndarray nor None for resample_x() in '+\
                'PedestrianIntentionApplicationInterface.')

    def predict_trajectories(self, x_obs, intention, truncated=False):
        """
        Predict pedestrian trajectories given observation and intention from particles.

        Inputs:
            - x_obs: numpy. :math:`(obs_seq_len, 2)`.
            - intention: numpy. :math:`(num_particles,)` Intention hypotheses for all particles. 
            e.g. for num_intentions=3, num_particles_per_intention=5, intention = 
            array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).
            - truncated: True means we use prediction function in propagate_x() or we only want 
            predicted trajectories for evaluation. False means we are performing long term 
            prediction until the goal position.

        Updated:
            - None
            
        Outputs:
            - x_est: If truncated=True, numpy. :math:`(num_particles, T_f, 2)`. 
            Otherwise a numpy array of objects. (num_particles, ), where one object is numpy. 
            :math:`(variable_t, 2)`. The predicted trajectories of particles. 
        """
        if self.method == 'wlstm':
            x_est = self.predict_warp_lstm(x_obs, intention, truncated=truncated)
        elif self.method == 'ilm':
            x_est = self.predict_intention_aware_linear_model(x_obs, intention, truncated=truncated)
        else:
            raise RuntimeError('Wrong method for PedestrianIntentionApplicationInterface.')
        return x_est


    def predict_intention_aware_linear_model(self, x_obs, intention, truncated=False):
        """
        Predict trajectories given observation and intention hypotheses using intention aware 
        linear model.

        Inputs:
            - x_obs: numpy. The observed trajectory. Can be either :math:`(t-T_f, 2)` or 
            math:`(t, 2)`, depending on whether we do propagate_x() or long term prediction.
            - intention: numpy. :math:`(num_particles,)` Intention hypotheses for all particles. 
            e.g. for num_intentions=3, num_particles_per_intention=5, intention = 
            array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).
            - truncated: True means we use prediction function in propagate_x() or we only want 
            predicted trajectories for evaluation. False means we are performing long term 
            prediction until the goal position.
            
        Updated:
            - None
        
        Outputs:
            - x_est: If truncated=True, numpy. :math:`(num_particles, T_f, 2)`. 
            Otherwise a numpy array of objects. (num_particles, ), where one object is numpy. 
            :math:`(variable_t, 2)`. The predicted trajectories of particles.
        """
        num_particles = len(intention)
        if truncated:
            x_est = np.empty((num_particles, self.Tf, 2))
        else:
            x_est = np.empty(num_particles).astype(object)
        heuristic_num_steps, goal_position_samples = self._intention_aware_linear_model_heuristic(x_obs, intention)
        survived_intention_indices = np.unique(intention)
        for i in survived_intention_indices:
            goal_position_samples_i = goal_position_samples[intention==i]
            prediction_samples_i_ilm = (np.linspace(x_obs[-1], goal_position_samples_i, \
                num=heuristic_num_steps[i])).transpose(1,0,2) # (num_particles_of_that_intention, heuristic_steps, 2)
            if truncated:
                # do not include x_obs[-1]
                x_est[intention==i] = prediction_samples_i_ilm[:,1:self.Tf+1] # (num_particles_of_that_intention, Tf, 2)
            else:
                x_est[np.where(intention==i)[0]]=list(prediction_samples_i_ilm)
        return x_est

    def predict_warp_lstm(self, x_obs, intention, truncated=False):
        """
        Predict trajectories given observation and intention hypotheses using Warp LSTM.

        Inputs:
            - x_obs: numpy. The observed trajectory. Can be either :math:`(t-T_f, 2)` or 
            math:`(t, 2)`, depending on whether we do propagate_x() or long term prediction.
            - intention: numpy. :math:`(num_particles,)` Intention hypotheses for all particles. 
            e.g. for num_intentions=3, num_particles_per_intention=5, intention = 
            array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).
            - truncated: True means we use prediction function in propagate_x() or we only want 
            predicted trajectories for evaluation. False means we are performing long term 
            prediction until the goal position.
            
        Updated:
            - None
        
        Outputs:
            - x_est: If truncated=True, numpy. :math:`(num_particles, T_f, 2)`. 
            Otherwise a numpy array of objects. (num_particles, ), where one object is numpy. 
            :math:`(variable_t, 2)`. The predicted trajectories of particles.
        """
        num_particles = len(intention)
        if truncated:
            x_est = np.empty((num_particles, self.Tf, 2))
        else:
            x_est = np.empty(num_particles).astype(object)
        heuristic_num_steps, goal_position_samples = self._intention_aware_linear_model_heuristic(x_obs, intention)
        survived_intention_indices = np.unique(intention)
        for i in survived_intention_indices:
            # different intentions may have different observation ratios, and thus need to use different wlstms.
            goal_position_samples_i = goal_position_samples[intention==i]
            if len(goal_position_samples_i) == 0:
                raise RuntimeError('The intention is not found in particles.')
            prediction_samples_i = (np.linspace(x_obs[-1], goal_position_samples_i, \
                num=heuristic_num_steps[i])).transpose(1,0,2) # (num_particles_of_that_intention, heuristic_steps, 2)
            prediction_samples_i_ilm = prediction_samples_i[:,1:] # (num_particles_of_that_intention, heuristic_steps-1, 2) # pred_seq_len = heuristic_steps-1
            x_obs_i = x_obs * np.ones((prediction_samples_i_ilm.shape[0],1,1)) # (num_particles_of_that_intention, obs_seq_len, 2)
            obs_seq_len, pred_seq_len = x_obs_i.shape[1], prediction_samples_i_ilm.shape[1]
            observation_ratio = float(obs_seq_len)/float(obs_seq_len+pred_seq_len)
            if observation_ratio <= 0.125:
                model_index = 0
            elif observation_ratio <= 0.375:
                model_index = 1
            elif observation_ratio <= 0.625:
                model_index = 2
            else:
                model_index = 3
            sample_base = np.concatenate((x_obs_i, prediction_samples_i_ilm), axis=1) # (num_particles_of_that_intention, obs_seq_len + pred_seq_len, 2)
            sample_loss_mask = np.concatenate((np.zeros((prediction_samples_i_ilm.shape[0], obs_seq_len, 1)),
                np.ones((prediction_samples_i_ilm.shape[0], pred_seq_len, 1))), axis=1) # (num_particles_of_that_intention, obs_seq_len + pred_seq_len, 1)
            sample_length = np.ones(prediction_samples_i_ilm.shape[0]) * (obs_seq_len+pred_seq_len)
            sample_base, sample_loss_mask, sample_length = \
                torch.from_numpy(sample_base).float().to(self.device), \
                torch.from_numpy(sample_loss_mask).float().to(self.device), \
                torch.from_numpy(sample_length).int().to(self.device)
            sample_improved = self.model[model_index](sample_base, sample_loss_mask, sample_length)
            prediction_samples_i_wlstm = sample_improved[:,-pred_seq_len:].detach().to('cpu').numpy() # (num_particles_of_that_intention, heuristic_steps-1, 2)
            if truncated:
                x_est[intention==i] = prediction_samples_i_wlstm[:,:self.Tf] # (num_particles_of_that_intention, Tf, 2)
            else:
                x_est[np.where(intention==i)[0]]=list(prediction_samples_i_wlstm)
        return x_est
 
    def _intention_aware_linear_model_heuristic(self, x_obs, intention):
        """
        Heuristic part of intention aware linear model. Repeatedly used in both intention aware 
        linear model prediction and Warp LSTM prediction.
        
        Inputs:
            - x_obs: numpy. The observed trajectory. Can be either :math:`(t-T_f, 2)` or 
            math:`(t, 2)`, depending on whether we do propagate_x() or long term prediction.
            - intention: numpy. :math:`(num_particles,)` Intention hypotheses for all particles. 
            e.g. for num_intentions=3, num_particles_per_intention=5, intention = 
            array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).
            
        Updated:
            - None
        
        Outputs:
            - heuristic_num_steps: numpy. :math:`(num_intentions,)`. Approximately how many steps are 
            required (including the last observed step) to reach the desired goal region (intention).
            - goal_position_samples: numpy. :math:`(num_particles, 2)` samples of goal positions.
        """
        if len(x_obs) <= 1:
            raise RuntimeError("The time steps of x_obs has to be at least 2.")
        heuristic_distance = np.linalg.norm(self.pedestrian_intention_sampler.intention_center_coordinates \
            - x_obs[-1], axis=1) # np (num_intentions,)
        mean_vel_mag = np.mean(np.linalg.norm(x_obs[1:]-x_obs[:-1], axis=1))
        step_num_noise = np.random.randint(2, 10, size=self.pedestrian_intention_sampler.num_intentions) # heuristic
        heuristic_num_steps = (heuristic_distance/mean_vel_mag).astype(int)+step_num_noise
        heuristic_num_steps[heuristic_num_steps<self.Tf+1] = self.Tf+1
        goal_position_samples = self.pedestrian_intention_sampler.sampling_goal_positions(intention)
        return heuristic_num_steps, goal_position_samples
    
    def get_num_intentions(self):
        """
        Return number of potential intentions.
        
        Inputs:
            - None
            
        Updated:
            - None
        
        Outputs:
            - num_intentions: Number of potential intentions in the scene.
        """
        return self.pedestrian_intention_sampler.num_intentions


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