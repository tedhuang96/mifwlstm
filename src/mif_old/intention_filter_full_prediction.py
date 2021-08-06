import numpy as np
import matplotlib.pyplot as plt

from src.mif.intention_sampler import IntentionSampler
from src.mif.intention_particle import IntentionParticle
from src.mif.sample_predictor import SamplePredictor

r"""
infos include full prediction in intention_filter_full_prediction,
where full prediction means prediction from last observed position to the intention.
This will increase a lot the size of the filtering data.
"""
def IntentionFilter(sample_true, particle_num_per_intent, num_tpp, tau, pred_func, step_per_update=2, display_on=True, mutation_on=False, warmup_step=10):
    percentage_hist = []
    intention_dist_hist = []
    particle_intention_hist = []
    long_term_pred_hist, long_term_obs_hist, long_term_true_hist = [], [], []
    full_pred_hist, full_true_hist = [], []
    i_sampler = IntentionSampler()
    i_particle = IntentionParticle(i_sampler, particle_num_per_intent=particle_num_per_intent, num_tpp=num_tpp)
    correct_intention = i_sampler.belong2intent(sample_true[-1])
    if display_on:
        print('Correct intention: ', correct_intention)
    for last_obs_index in range(num_tpp+warmup_step, len(sample_true)-num_tpp, step_per_update): # not include 0.
        # print('last obs index: ', last_obs_index)
        x_pred_true = sample_true[last_obs_index-num_tpp+1:last_obs_index+1].numpy()
        x_obs = sample_true[:last_obs_index-num_tpp+1].numpy()
        x_true_remained = sample_true[last_obs_index+1:].numpy()
        percentage_curr = float(last_obs_index/len(sample_true))
        ## truncated prediction
        short_pred = i_particle.predict(x_obs, pred_func) # short_pred # np # (340, 12, 2)
        ## particle weight update and importance resampling
        i_particle.update_weight(x_pred_true, tau=tau)
        i_particle.resample()
        ## mutation
        if mutation_on:
            i_particle.mutate()
        ## intention prob. dist.
        _, intention_weight_dist = i_particle.particle_weight_intention_prob_dist()
        particle_intention = i_particle.intention
        sample_full_pred = i_particle.predict_till_end(sample_true[:last_obs_index+1].numpy(), pred_func)
        long_pred = []
        for xy_pred in sample_full_pred:
            if len(xy_pred) >= num_tpp:
                long_pred.append(xy_pred[:num_tpp])
        long_pred = np.stack(long_pred, axis=0) # (sample_size, num_tpp, 2)
        x_obs, x_gt, x_pred = x_pred_true, x_true_remained[:num_tpp], long_pred
        x_gt_full, x_pred_full =  x_true_remained, sample_full_pred
        percentage_hist.append(percentage_curr)
        intention_dist_hist.append(intention_weight_dist)
        particle_intention_hist.append(particle_intention)
        long_term_pred_hist.append(x_pred)
        long_term_true_hist.append(x_gt)
        long_term_obs_hist.append(x_obs)
        full_pred_hist.append(x_pred_full)
        full_true_hist.append(x_gt_full)
        if display_on:
            print('The intention weight_dist: ', intention_weight_dist)
            print('The percentage: ', percentage_curr)
            for xy_pred_short in short_pred:            
                plt.plot(xy_pred_short[:,0], xy_pred_short[:,1], 'k-')
            for sample_pred in x_pred:
                plt.plot(sample_pred[:,0], sample_pred[:,1], 'c-')
            plt.plot(x_obs[:, 0], x_obs[:, 1], 'g-')
            plt.plot(x_gt[:, 0],x_gt[:, 1], 'r-')
            plt.xlim(-1, 16)
            plt.ylim(-1, 14)
            plt.show()
    infos = [correct_intention, percentage_hist, intention_dist_hist, particle_intention_hist, long_term_pred_hist, long_term_true_hist,\
            long_term_obs_hist, full_pred_hist, full_true_hist]
    return infos

def filtering_data_format():
    """
        x_obs # np # (num_tpp, 2)
        x_gt # np # (num_tpp, 2)
        x_pred # np # (sample_num, num_tpp, 2)
        x_gt_full # np (remained_step, 2)
        x_pred_full # np # (340, )
            - sample # np # (time_to_go_estimate, 2)
    """
    pass
