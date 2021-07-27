from pathhack import pkg_path
from os.path import join
import pickle
import numpy as np
import torch
import argparse
import time
import scipy.stats as stats
np.random.seed(0)

def eval_traj():
    args = arg_parse()
    print()
    print()
    print('----------------------------------------------------------------------')
    print('arguments')
    print(args)

    logdir = join(pkg_path, 'results', 'mif', args.pred_func+'_tau_'+str(args.tau))
    if not args.mutation_on:
        logdir = logdir + '_no_mutate'
    print('folder: ', logdir)

    infos_list_all = []
    for i in range(1, 11): # (1, ..., 10)
        filename = 'infos_list_'+str(i)+'.p'
        filepath = join(logdir, filename)
        with open(filepath, 'rb') as f:
            infos_list = pickle.load(f)
            print(filepath+' is loaded.')
        infos_list_all = infos_list_all + infos_list

    print(len(infos_list_all))
    aoe_list, moe_list, nll_list, aoen_list, moen_list, nlln_list = [], [], [], [], [], [] # n means noise
    aoe_mean_list, moe_mean_list, aoen_mean_list, moen_mean_list = [], [], [], []
    foe_list, foen_list, foe_mean_list, foen_mean_list = [], [], [], []
    start_time = time.time()
    for i, infos in enumerate(infos_list_all):
        if (i+1) % 500 == 0:
            print('The {0}th sample in test data'.format(i+1))
        [correct_intention, percentage_hist, intention_dist_hist, particle_intention_hist, long_term_pred_hist, long_term_true_hist,\
                long_term_obs_hist] = infos
        for ts in range(len(long_term_obs_hist)):
            x_obs = long_term_obs_hist[ts]
            x_gt = long_term_true_hist[ts]
            x_pred = long_term_pred_hist[ts]
            intention_dist = intention_dist_hist[ts]
            particle_intention = particle_intention_hist[ts]
            # top 3 intentions # use top 3 or top 1 trajectory
            top_intentions = np.flip(np.argsort(intention_dist))[:args.num_top_intentions]
            valid_samples = []
            for i, intention in enumerate(top_intentions):
                sample_idx = np.where(particle_intention==intention)[0]
                samples = x_pred[sample_idx]
                valid_samples.append(samples)

            valid_samples = np.concatenate(valid_samples, axis=0) # (valid_sample_num, num_tpp, 2)
            valid_x_gt = np.ones_like(valid_samples) * x_gt
            loss_mask = np.ones(valid_samples.shape[:2])[:,:,np.newaxis]
            oe = offset_error(torch.tensor(valid_x_gt), torch.tensor(valid_samples), torch.tensor(loss_mask)).numpy() # (valid_sample_num, num_tpp)
            aoe, moe, foe = np.mean(oe, axis=1), np.max(oe, axis=1), oe[:, -1]
            min_aoe, min_moe, min_foe = np.min(aoe), np.min(moe), np.min(foe)
            mean_aoe, mean_moe, mean_foe = np.mean(aoe), np.mean(moe), np.mean(foe)
            nll = negative_log_likelihood(x_gt, valid_samples)
            aoe_mean_list.append(mean_aoe)
            moe_mean_list.append(mean_moe)
            foe_mean_list.append(mean_foe)

            aoe_list.append(min_aoe)
            moe_list.append(min_moe)
            foe_list.append(min_foe)
            nll_list.append(nll)

            valid_samples += np.random.rand(*valid_samples.shape) * 0.05
            nll = negative_log_likelihood(x_gt, valid_samples)
            oe = offset_error(torch.tensor(valid_x_gt), torch.tensor(valid_samples), torch.tensor(loss_mask)).numpy() # (valid_sample_num, num_tpp)
            aoe, moe, foe = np.mean(oe, axis=1), np.max(oe, axis=1), oe[:, -1]
            min_aoe, min_moe, min_foe = np.min(aoe), np.min(moe), np.min(foe)
            mean_aoe, mean_moe, mean_foe = np.mean(aoe), np.mean(moe), np.mean(foe)
            aoen_mean_list.append(mean_aoe)
            moen_mean_list.append(mean_moe)
            foen_mean_list.append(mean_foe)
            aoen_list.append(min_aoe)
            moen_list.append(min_moe)
            foen_list.append(min_foe)
            nlln_list.append(nll)
    
    aoe_final, moe_final, foe_final, nll_final = np.mean(aoe_list), np.mean(moe_list), np.mean(foe_list), np.mean(nll_list)
    aoen_final, moen_final, foen_final, nlln_final = np.mean(aoen_list), np.mean(moen_list), np.mean(foen_list), np.mean(nlln_list)
    aoe_mean_final, moe_mean_final, foe_mean_final, aoen_mean_final, moen_mean_final, foen_mean_final = \
        np.mean(aoe_mean_list), np.mean(moe_mean_list), np.mean(foe_mean_list), np.mean(aoen_mean_list), np.mean(moen_mean_list), np.mean(foen_mean_list)
    print('number of top intentions: {0}'.format(args.num_top_intentions))
    print('aoe_final:{0:.3f}, moe_final:{1:.3f}, foe_final: {2:.3f}, nll_final:{3:.3f}'.format(aoe_final, moe_final, foe_final, nll_final))
    print('aoe_mean_final:{0:.3f}, moe_mean_final:{1:.3f}, foe_mean_final: {2:.3f}'.format(aoe_mean_final, moe_mean_final, foe_mean_final))
    print('aoen_final:{0:.3f}, moen_final:{1:.3f}, foen_final: {2:.3f}, nlln_final:{3:.3f}'.format(aoen_final, moen_final, foen_final, nlln_final))
    print('aoen_mean_final:{0:.3f}, moen_mean_final:{1:.3f},  foen_mean_final: {2:.3f}'.format(aoen_mean_final, moen_mean_final, foen_mean_final))
    print('time to filter: {0:.2f} min'.format((time.time()-start_time)/60.))
    print('----------------------------------------------------------------------')




def negative_log_likelihood(x_gt, x_pred, eps=1e-6):
    """
    masked offset error for global coordinate data.
    inputs:
        - x_gt: ground truth future data. tensor. size: (seq_len, 2)
        - x_pred: future data prediction. tensor. size: (batch, seq_len, 2) 
    outputs:
        - oe: (batch, seq_len)
    """
    seq_len = x_gt.shape[0]
    nll_list = []
    for tt in range(seq_len):
        x_pred_t = x_pred[:, tt] # (batch, 2)
        try:
            kernel_t = stats.kde.gaussian_kde(x_pred_t.T) # (2, batch)
            nll_t = -np.log(kernel_t(x_gt[tt])+eps) #(1, ) # prevent log(0)
            nll_list.append(nll_t)
        except np.linalg.LinAlgError as err:
            pass

    nll_np = np.array(nll_list)
    nll_mean = np.mean(nll_np)
    return nll_mean

def offset_error(x_gt, x_pred, loss_mask):
    """
    masked offset error for global coordinate data.
    inputs:
        - x_gt: ground truth future data. tensor. size: (batch, seq_len, 2)
        - x_pred: future data prediction. tensor. size: (batch, seq_len, 2)
        - loss_mask: 0-1 mask on prediction range. size: (batch, seq_len, 1)
          We now want mask like [0, 0, ..., 1, 1, ..,1, 0., 0] to work. And it works. 
    outputs:
        - oe: (batch, seq_len)
    """
    oe = (((x_gt - x_pred) ** 2.).sum(dim=2)) ** (0.5) # (batch, seq_len)
    oe_masked = oe * loss_mask.squeeze(dim=2) # (batch, seq_len)
    return oe_masked
    
def arg_parse():
    parser = argparse.ArgumentParser()

    ##### rebil #####
    # Dataset Options
    parser.add_argument('--dataset_ver', default=0, type=int)
    # Optimization Options
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_epochs', default=200, type=int)
    # Model Options
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--end_mask', action='store_true')
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--num_lstms', default=1, type=int)
    parser.add_argument('--embedding_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    # Evaluation Options
    parser.add_argument('--eval_model_saved_epoch', default=None, type=int)
    ##### intention filter #####
    parser.add_argument('--filter_test_data', default=1, type=int,\
        help="1-10 represents the 1/10 till 10/10 in dataset.")
    parser.add_argument('--particle_num_per_intent', default=10, type=int,\
        help="10, 30, 50, 100.")
    parser.add_argument('--num_tpp', default=20, type=int, help="12, 20")
    parser.add_argument('--step_per_update', default=2, type=int, help="1, 2")
    parser.add_argument('--tau', default=10., type=float, help="1., 10.")
    parser.add_argument('--pred_func', default='rebil', type=str, help='rebil or ilm.')
    parser.add_argument('--mutation_on', action='store_true')
    parser.add_argument('--display_on', action='store_true')
    parser.add_argument('--num_top_intentions', default=3, type=int, help='1, 3.')
    return parser.parse_args()



if __name__ == "__main__":
    eval_traj()