from os.path import join
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import padding, unpadding, padding_mask, \
    batch_iter_no_shuffle, load_warplstm_model


def visualize_warplstm(
    zipped_data_test,
    logdir,
    visual_batch_size=1,
    visual_num_images=10,
    bidirectional=True,
    end_mask=False,
    num_layers=1,
    num_lstms=3,
    num_epochs=200,
    embedding_size=128,
    hidden_size=128,
    dropout=0.,
    saved_model_epoch=None,
    device='cuda:0',
):
    ##### Unzip Data and Load Model #####
    traj_base_test, traj_true_test, traj_loss_mask_test = zipped_data_test
    model = load_warplstm_model(
        logdir,
        saved_model_epoch,
        num_epochs,
        embedding_size,
        hidden_size,
        num_layers,
        num_lstms,
        dropout,
        bidirectional,
        end_mask,
        device,
    )
    ##### Visualization #####
    with torch.no_grad():
        sample_indices = np.random.randint(len(traj_base_test), size=visual_num_images)
        print(sample_indices)
        traj_base_test_sampled = [traj_base_test[i] for i in sample_indices]
        traj_true_test_sampled = [traj_true_test[i] for i in sample_indices]
        traj_loss_mask_test_sampled = [traj_loss_mask_test[i] for i in sample_indices]
        for i, (samples_base, samples_true, samples_loss_mask) in \
            enumerate(batch_iter_no_shuffle(
                traj_base_test_sampled, 
                traj_true_test_sampled,
                traj_loss_mask_test_sampled,
                batch_size=visual_batch_size,
            )):
            sb, sl = padding(samples_base)
            st, _ = padding(samples_true)
            sm_pred, _ = padding(samples_loss_mask) # (64, 526, 1)å
            sm_all = padding_mask(sl)
            sm_obs = sm_all - sm_pred
            sb, sl, st, sm_pred, sm_all = sb.to(device), sl.to(device), st.to(device), sm_pred.to(device), sm_all.to(device)
            sb_improved = model(sb, sm_pred, sl)
            ## plot
            sample_pred_unpadded = unpadding(sb_improved.to('cpu'), sl)
            fig, ax = plt.subplots()
            colors = 'ymcr'
            color_idx = 0
            for sample_base, sample_pred, sample_true, sample_loss_mask_obs in zip(samples_base, sample_pred_unpadded, samples_true, sm_obs):
                obs_len = sample_loss_mask_obs.sum().int()
                ax.plot(sample_true[:, 0], sample_true[:, 1], 'g', lw=5) # true
                ax.plot(sample_pred[:obs_len, 0], sample_pred[:obs_len, 1], 'b', lw=3) # obs
                ax.plot(sample_base[obs_len:, 0], sample_base[obs_len:, 1], 'k',lw=3) # baseline
                ax.plot(sample_pred[obs_len:, 0], sample_pred[obs_len:, 1], colors[color_idx],lw=3) # pred
                ax.set(xlim=(-1, 16), ylim=(-1, 14))
                color_idx = color_idx + 1
            ## save plots and traj data.
            fig_name = join(logdir, 'visual_'+str(i)+'.png')
            fig.savefig(fig_name)
            # with open(join(logdir, 'vis_'+str(i)+'.p'), 'wb') as f:
            #     pickle.dump([samples_base, samples_true, samples_loss_mask], f)
            #     print(join(logdir, 'vis_'+str(i)+'.p')+' is dumped.')