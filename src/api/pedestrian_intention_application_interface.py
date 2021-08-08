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
            - pkg_path: absolute path of the package.
            - device: 'cuda:0' or 'cpu'.

        Updated:
            - self.application: 'pedestrian2D'
            - self.method
            - self.scene
            - self.device
        
        Outputs:
            - None
        """
        super(PedestrianIntentionApplicationInterface, self).__init__('pedestrian2D')
        self.method = method
        self.scene = scene
        self.device = device
        if self.method == 'wlstm':
            self.model = load_warplstm_model_by_arguments(args, pkg_path, self.device)
        elif self.method == 'ilm':
            self.model = None
        else:
            raise RuntimeError('Wrong method input for PedestrianIntentionApplicationInterface.')
        if self.scene = 'edinburgh':
            pass
        return



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
            - self.intention_bottomleft_coordinates
            - self.scene_xlim
            - self.scene_ylim
        
        Outputs:
            - None
        """
        self.intention_width = intention_width
        self.num_intentions = num_intentions
        self.intention_bottomleft_coordinates = intention_bottomleft_coordinates
        self.scene_xlim, self.scene_ylim = scene_xlim, scene_ylim
        return

        
    def visualize_intention(self, traj_end_pos=None):
        """
        Visualize the scene with intentions and end positions of trajectories if any.

        Inputs:
            - traj_end_pos: None or numpy. :math:`(num_trajectories, 2)` End positions of trajectories.

        Updated:
            - None
        
        Outputs:
            - fig
            - ax
        """
        fig, ax = plt.subplots()
        if traj_end_pos is not None:
            ax.plot(traj_end_pos[:, 0], traj_end_pos[:, 1], 'b.')
        for intent_bottom_left in self.intention_bottomleft_coordinates:
            rect = patches.Rectangle(intent_bottom_left, self.intention_width, self.intention_width,linewidth=1,edgecolor='k',facecolor='none',zorder=3)
            ax.add_patch(rect)
        ax.set(xlim=self.scene_xlim, ylim=self.scene_ylim)
        ax.axis('scaled')
        return fig, ax
        
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
        bool_flag = ((vec_bl_to_pos[:, 0] > 0) * (vec_bl_to_pos[:, 0] < self.intention_width)) * \
            ((vec_bl_to_pos[:, 1] > 0) * (vec_bl_to_pos[:, 1] < self.intention_width))
        if len(np.where(bool_flag)[0]) == 0:
            return None
        else:
            return np.where(bool_flag)[0][0] # index of intention
        # ! Zhe Pause Here
    
    def idx2intent_sampling(self, intent_idx):
        intent_bf_samples = self.intention_bottomleft_coordinates[intent_idx]
        sample_noise = np.random.uniform(low=0., high=self.intention_width, \
                                         size=intent_bf_samples.shape)
        intent_samples = intent_bf_samples + sample_noise
        return intent_samples
    
    def uniform_sampling(self, sample_num_per_intent=20):
        sample_noise = np.random.uniform(low=0., high=self.intention_width, \
                                         size=(self.num_intentions, sample_num_per_intent, 2))
        intent_samples = sample_noise + self.intention_bottomleft_coordinates[:, np.newaxis, :]
        intent_samples = intent_samples.reshape(-1, 2)
        return intent_samples