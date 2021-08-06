import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class IntentionSampler:
    def __init__(self):
        self.intent_wid = 1.5
        self.intent_bottom_left_coordinates = self.generate_bottom_left_positions()
        self.intent_num = len(self.intent_bottom_left_coordinates)
        
    def visualize_intent(self, traj_end_pos=None):
        fig, ax = plt.subplots()
        if traj_end_pos is not None:
            ax.plot(traj_end_pos[:, 0], traj_end_pos[:, 1], 'b.')
        for intent_bottom_left in self.intent_bottom_left_coordinates:
            rect = patches.Rectangle(intent_bottom_left, self.intent_wid, self.intent_wid,linewidth=1,edgecolor='k',facecolor='none',zorder=3)
            ax.add_patch(rect)
        ax.set(xlim=(-1, 16), ylim=(-1, 14))
        ax.axis('scaled')
        return fig, ax
        
    def belong2intent(self, point):
        vec_bf2point = point - self.intent_bottom_left_coordinates
        bool_flag = ((vec_bf2point[:, 0] > 0) * (vec_bf2point[:, 0] < 1.5)) * \
            ((vec_bf2point[:, 1] > 0) * (vec_bf2point[:, 1] < 1.5))
        if len(np.where(bool_flag)[0]) == 0:
            return None
        else:
            return np.where(bool_flag)[0][0] # index of intent
    
    def idx2intent_sampling(self, intent_idx):
        intent_bf_samples = self.intent_bottom_left_coordinates[intent_idx]
        sample_noise = np.random.uniform(low=0., high=self.intent_wid, \
                                         size=intent_bf_samples.shape)
        intent_samples = intent_bf_samples + sample_noise
        return intent_samples
    
    def uniform_sampling(self, sample_num_per_intent=20):
        sample_noise = np.random.uniform(low=0., high=self.intent_wid, \
                                         size=(self.intent_num, sample_num_per_intent, 2))
        intent_samples = sample_noise + self.intent_bottom_left_coordinates[:, np.newaxis, :]
        intent_samples = intent_samples.reshape(-1, 2)
        return intent_samples
        
    def generate_bottom_left_positions(self):
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
        x_bottom_left_np = np.stack(x_bottom_left_list, axis=0)
        return x_bottom_left_np


if __name__ == "__main__":
    # test
    i_sampler = IntentionSampler()
    intentions = np.arange(3,19)
    intent_samples = i_sampler.idx2intent_sampling(intentions)
    fig, ax = i_sampler.visualize_intent(traj_end_pos=intent_samples)
    print('belong2intent test: ', i_sampler.belong2intent(intent_samples[-1]))