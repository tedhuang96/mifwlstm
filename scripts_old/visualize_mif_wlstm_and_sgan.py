from pathhack import pkg_path
from os.path import join
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
ax_x_lim, ax_y_lim = (3, 14), (3.5, 9.5)
x_ticks = [4, 8, 12]
y_ticks = [4, 8]
plt_alpha = 0.3

def visualize_plt_data_gan():
    hues = [[215,48,39], [244,109,67], [253,174,97], [171,217,233], [116,173,209], [69,117,180]]
    for i in range(len(hues)):
        hues[i] = np.array(hues[i])/255.
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    result_filename = 'plt_data_sgan_tau_10.p'
    filepath = join(pkg_path, 'results/visual', result_filename)
    with open(filepath, 'rb') as f:
        plt_data_for_paper = pickle.load(f)
        print(filepath+' is loaded.')
    x_obs, x_gt, _, x_pred = plt_data_for_paper
    result_filename = 'edin3_pred.p'
    filepath = join(pkg_path, 'results/visual', result_filename)
    with open(filepath, 'rb') as f:
        x_pred = pickle.load(f)
        print(filepath+' is loaded.')
    for sample_pred in x_pred:
        sample_pred = sample_pred.to('cpu')
        sample_pred = np.concatenate((x_obs[-1:,:], sample_pred), axis=0)
        ax.plot(sample_pred[:,0], sample_pred[:,1], c=hues[4], alpha=plt_alpha)
    x_gt = np.concatenate((x_obs[-1:,:], x_gt), axis=0)
    ax.plot(x_obs[:, 0], x_obs[:, 1], 'k-', label='')
    ax.plot(x_gt[:, 0],x_gt[:, 1], 'k--')
    ax.set_aspect('equal',adjustable='box')
    ax.set(xlim=ax_x_lim, ylim=ax_y_lim)
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    fig_filename = 'plt_samples_gan.pdf'
    fig.savefig(join(pkg_path, 'results/visual', fig_filename),bbox_inches='tight')
    print(join(pkg_path, 'results/visual', fig_filename)+' is created.')

   



def visualize_plt_data_mif_wlstm(config_name):
    """
    inputs:
        - config_name: 'tau_1', 'tau_10'
    """
    hues = [[215,48,39], [244,109,67], [253,174,97], [171,217,233], [116,173,209], [69,117,180]]
    for i in range(len(hues)):
        hues[i] = np.array(hues[i])/255.
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    if config_name == 'tau_1':
        result_filename = 'plt_data_sgan_tau_1.p'
    elif config_name == 'tau_10':
        result_filename = 'plt_data_sgan_tau_10.p'
    filepath = join(pkg_path, 'results/visual', result_filename)
    with open(filepath, 'rb') as f:
        plt_data_for_paper = pickle.load(f)
        print(filepath+' is loaded.')
    x_obs, x_gt, valid_samples, x_pred = plt_data_for_paper
    for sample_pred in x_pred:
        sample_pred = sample_pred - sample_pred[0:1,:] + x_obs[-1:,:] # (20, 2)
        sample_pred_last = sample_pred[-1:] - sample_pred[-2:-1] + sample_pred[-1:] # (1, 2)
        sample_pred = np.concatenate([sample_pred[1:], sample_pred_last], axis=0)
        sample_pred = np.concatenate((x_obs[-1:,:], sample_pred), axis=0)
        ax.plot(sample_pred[:,0], sample_pred[:,1], c=hues[4], alpha=plt_alpha) # 3
    for i in [1, 2]:
        samples = valid_samples[i]
        for sample_pred in samples:
            sample_pred = sample_pred - sample_pred[0:1,:] + x_obs[-1:,:] # (20, 2)
            sample_pred_last = sample_pred[-1:] - sample_pred[-2:-1] + sample_pred[-1:] # (1, 2)
            sample_pred = np.concatenate([sample_pred[1:], sample_pred_last], axis=0)
            sample_pred = np.concatenate((x_obs[-1:,:], sample_pred), axis=0)
            ax.plot(sample_pred[:,0], sample_pred[:,1], c=hues[2], alpha=plt_alpha) # 5
    samples = valid_samples[0]
    for sample_pred in samples:
        sample_pred = sample_pred - sample_pred[0:1,:] + x_obs[-1:,:] # (20, 2)
        sample_pred_last = sample_pred[-1:] - sample_pred[-2:-1] + sample_pred[-1:] # (1, 2)
        sample_pred = np.concatenate([sample_pred[1:], sample_pred_last], axis=0)
        sample_pred = np.concatenate((x_obs[-1:,:], sample_pred), axis=0)
        ax.plot(sample_pred[:,0], sample_pred[:,1], c=hues[0], alpha=plt_alpha)
    x_gt = np.concatenate((x_obs[-1:,:], x_gt), axis=0)
    ax.plot(x_obs[:, 0], x_obs[:, 1], 'k-', label='')
    ax.plot(x_gt[:, 0],x_gt[:, 1], 'k--')
    ax.set_aspect('equal',adjustable='box')
    ax.set(xlim=ax_x_lim, ylim=ax_y_lim)
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)
    if config_name == 'tau_1':
        fig_filename = 'plt_samples_tau_1.pdf'
    elif config_name == 'tau_10':
        fig_filename = 'plt_samples_tau_10.pdf'
    fig.savefig(join(pkg_path, 'results/visual', fig_filename),bbox_inches='tight')
    print(join(pkg_path, 'results/visual', fig_filename)+' is created.')



if __name__ == "__main__":
    config_name = 'tau_1'
    visualize_plt_data_mif_wlstm(config_name)
    config_name = 'tau_10'
    visualize_plt_data_mif_wlstm(config_name)
    visualize_plt_data_gan()