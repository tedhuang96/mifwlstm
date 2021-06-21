from pathhack import pkg_path
import sys
from os.path import isdir, join
import time
import pickle
import numpy as np
import argparse

import torch
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from src.utils import average_offset_error, max_offset_error, final_offset_error, batch_iter, \
    padding, unpadding, padding_mask, batch_iter_no_shuffle, \
    load_preprocessed_train_test_dataset, ilm
from src.wlstm.models import ReBiL
from src.wlstm.utils import load_rebil_model

np.random.seed(5)

def train_rebil(zipped_data_train_test, writer, logdir, lr=1e-4, batch_size=32, bidirectional=True, end_mask=False, num_layers=1, \
                num_lstms=3, num_epochs=100, embedding_size=256, hidden_size=256, save_epochs=10, activation=None, batch_norm=True, \
                dropout=0., clip_grad_norm=10., device='cuda:0'):
    ##### Unzip Data #####
    traj_base_train, traj_true_train, traj_loss_mask_train, \
        traj_base_test, traj_true_test, traj_loss_mask_test = zipped_data_train_test
    ##### Initialize Model & Optimizer ##### 
    model = ReBiL(embedding_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, \
        num_lstms=num_lstms, bidirectional=bidirectional, end_mask=end_mask, device=device).to(device)
    model_params = list(model.parameters())
    for lstm in model.lstms:
        model_params = model_params + list(lstm.parameters())
    optimizer = torch.optim.Adam(model_params, lr=lr)
    print()
    print('MODEL INITIALIZATION')
    print('learning rate: ', lr)
    print('model architecture: ')
    print(model)
    print(model.lstms)
    print()

    ##### Training #####
    print()
    print('TRAINING STARTED')
    print('# EPOCHS: ', num_epochs)
    train_loss_task, test_loss_task, train_aoe_task, test_aoe_task = [],[],[],[]
    for epoch in range(1, num_epochs+1):
        ## train ##
        epoch_start_time = time.time()
        train_sample_num_epoch = []
        train_loss_epoch = []
        train_results_epoch = []
        for samples_base, samples_true, samples_loss_mask in batch_iter(traj_base_train, traj_true_train, traj_loss_mask_train, batch_size=batch_size):
            optimizer.zero_grad()
            sb, sl = padding(samples_base)
            st, _ = padding(samples_true)
            sm_pred, _ = padding(samples_loss_mask)
            sm_all = padding_mask(sl)
            loss, sb_improved = model(sb.to(device), st.to(device), sm_pred.to(device), sm_all.to(device), sl.to(device))
            train_results_epoch.append(average_offset_error(st, sb_improved.to('cpu'), sm_pred))
            train_sample_num_epoch.append(len(samples_base))
            train_loss_epoch.append(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_params, clip_grad_norm)
            optimizer.step()
        training_epoch_period = time.time() - epoch_start_time
        training_epoch_period_per_sample = training_epoch_period/len(traj_base_train)
        ## eval ##
        with torch.no_grad():
            test_loss_epoch = []
            test_results_epoch = []
            test_sample_num_epoch = []
            for samples_base, samples_true, samples_loss_mask in batch_iter(traj_base_test, traj_true_test, traj_loss_mask_test, batch_size=batch_size):
                sb, sl = padding(samples_base)
                st, _ = padding(samples_true)
                sm_pred, _ = padding(samples_loss_mask) # (time_step, 1)
                sm_all = padding_mask(sl) 
                loss, sb_improved = model(sb.to(device), st.to(device), sm_pred.to(device), sm_all.to(device), sl.to(device))
                test_results_epoch.append(average_offset_error(st, sb_improved.to('cpu'), sm_pred))
                test_sample_num_epoch.append(len(samples_base))
                test_loss_epoch.append(loss)
            train_results_epoch = torch.tensor(train_results_epoch)
            test_results_epoch = torch.tensor(test_results_epoch)
            train_sample_num_epoch = torch.tensor(train_sample_num_epoch)
            train_loss_epoch = torch.tensor(train_loss_epoch)
            test_loss_epoch = torch.tensor(test_loss_epoch)
            test_sample_num_epoch = torch.tensor(test_sample_num_epoch)
            # compute mean of loss and average offset error exactly
            train_results_epoch = (train_results_epoch * train_sample_num_epoch).sum()/train_sample_num_epoch.sum()
            test_results_epoch = (test_results_epoch * test_sample_num_epoch).sum()/test_sample_num_epoch.sum()
            train_loss_epoch = (train_loss_epoch * train_sample_num_epoch).sum()/train_sample_num_epoch.sum()
            test_loss_epoch = (test_loss_epoch * test_sample_num_epoch).sum()/test_sample_num_epoch.sum()
            print('Epoch: {0} | train loss: {1:.4f} | test loss: {2:.4f} | train aoe: {3:.4f} | test aoe: {4:.4f} | period: {5:.2f} sec | time per sample: {6:.4f} sec'\
                        .format(epoch, train_loss_epoch.item(), test_loss_epoch.item(), train_results_epoch.item(), test_results_epoch.item(), training_epoch_period, training_epoch_period_per_sample))
            train_loss_task.append(train_loss_epoch.item())
            test_loss_task.append(test_loss_epoch.item())
            train_aoe_task.append(train_results_epoch.item())
            test_aoe_task.append(test_results_epoch.item())
            ## write to tensorboard ##
            writer.add_scalars('loss', {'train': train_loss_epoch.item(), 'test': test_loss_epoch.item()}, epoch)
            writer.add_scalars('aoe', {'train': train_results_epoch.item(), 'test': test_results_epoch.item()}, epoch)
            writer.add_scalars('loss', {'train': train_loss_epoch.item(), 'test': test_loss_epoch.item()}, epoch)
            writer.add_scalars('aoe', {'train': train_results_epoch.item(), 'test': test_results_epoch.item()}, epoch)
            ## save models ##
            if epoch % save_epochs == 0:
                model_filename = join(logdir, time.strftime('%Y_%m_%d-%H_%M_%S')+'-epoch_'+str(epoch)+'.pt')
                # parameters of lstms need to be manually saved.
                lstms_dict = []
                for res_lstm_component in model.lstms:
                    lstms_dict.append(res_lstm_component.state_dict())
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'lstms_dict': lstms_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_task,
                    'val_loss': test_loss_task,
                    'train_aoe': train_aoe_task,
                    'val_aoe': test_aoe_task,      
                    }, model_filename)
                print(model_filename+' is saved.')
    print()


def eval_rebil(zipped_data_test, model_eval, logdir, args, device='cuda:0'):
    ##### Unzip Data #####
    traj_base_test, traj_true_test, traj_loss_mask_test = zipped_data_test

    ##### Evaluate Model Performance #####
    with torch.no_grad():
        test_aoe_epoch, test_moe_epoch, test_foe_epoch = [], [], []
        test_sample_num_epoch = []
        for samples_base, samples_true, samples_loss_mask in batch_iter(traj_base_test, traj_true_test, traj_loss_mask_test, batch_size=args.batch_size):
            sb, sl = padding(samples_base)
            st, _ = padding(samples_true)
            sm_pred, _ = padding(samples_loss_mask) # (time_step, 1)
            sm_all = padding_mask(sl)
            sb_improved = model_eval.inference(sb.to(device), sm_pred.to(device), sm_all.to(device), sl.to(device), device=device)
            test_aoe_epoch.append(average_offset_error(st, sb_improved.to('cpu'), sm_pred))
            test_moe_epoch.append(max_offset_error(st, sb_improved.to('cpu'), sm_pred))
            test_foe_epoch.append(final_offset_error(st, sb_improved.to('cpu'), sm_pred))
            test_sample_num_epoch.append(len(samples_base))
        test_aoe_epoch = torch.tensor(test_aoe_epoch)
        test_moe_epoch = torch.tensor(test_moe_epoch)
        test_foe_epoch = torch.tensor(test_foe_epoch)
        test_sample_num_epoch = torch.tensor(test_sample_num_epoch)
        test_aoe_epoch = (test_aoe_epoch * test_sample_num_epoch).sum()/test_sample_num_epoch.sum()
        test_moe_epoch = (test_moe_epoch * test_sample_num_epoch).sum()/test_sample_num_epoch.sum()
        test_foe_epoch = (test_foe_epoch * test_sample_num_epoch).sum()/test_sample_num_epoch.sum()
        print('Evaluate model: ')
        print('aoe: {0:.4f} | moe: {1:.4f} | foe: {2:.4f}'.format(test_aoe_epoch.item(), test_moe_epoch.item(), test_foe_epoch.item()))
        print()
    
    ##### Visualize Model Performance #####
    if args.eval_visual:
        visual_rebil(zipped_data_test, model_eval, logdir, args, device=device)
    
    return


def visual_rebil(zipped_data_test, model_eval, logdir, args, device='cuda:0'):
    ##### Unzip Data #####
    traj_base_test, traj_true_test, traj_loss_mask_test = zipped_data_test
    ##### Set Up Keyword In Figure Name #####
    ## saved_epoch is used to save the visualization files.
    if args.eval_model_saved_epoch is None:
        saved_epoch = args.num_epochs
    else:
        saved_epoch = args.eval_model_saved_epoch
    ##### Visualization #####
    with torch.no_grad():
        vis_batch_size = args.visual_batch_size
        idx = np.random.randint(len(traj_base_test)-50, size=100)
        print(idx)
        a = [traj_base_test[i] for i in idx]
        b = [traj_true_test[i] for i in idx]
        c = [traj_loss_mask_test[i] for i in idx]
        traj_base_test, traj_true_test, traj_loss_mask_test = a, b, c
        for i, (samples_base, samples_true, samples_loss_mask) in enumerate(batch_iter_no_shuffle(traj_base_test, traj_true_test, traj_loss_mask_test, batch_size=vis_batch_size)):
            if i == args.visual_num_images:
                break
            sb, sl = padding(samples_base)
            st, _ = padding(samples_true)
            sm_pred, _ = padding(samples_loss_mask) # (time_step, 1)
            sm_all = padding_mask(sl)
            sm_obs = sm_all - sm_pred 
            sb_improved = model_eval.inference(sb.to(device), sm_pred.to(device), sm_all.to(device), sl.to(device), device=device)
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
            fig_name = join(logdir, 'vis_'+str(i)+'-epoch_'+str(saved_epoch)+'.png')
            fig.savefig(fig_name)
            with open(join(logdir, 'vis_'+str(i)+'-epoch_'+str(saved_epoch)+'.p'), 'wb') as f:
                pickle.dump([samples_base, samples_true, samples_loss_mask], f)
                print(join(logdir, 'vis_'+str(i)+'-epoch_'+str(saved_epoch)+'.p')+' is dumped.')

def arg_parse():
    parser = argparse.ArgumentParser()
    # Dataset Options
    parser.add_argument('--dataset_ver', default=0, type=int)
    # Optimization Options
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--clip_grad_norm', default=10., type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    # Model Options
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--end_mask', action='store_true')
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--num_lstms', default=3, type=int)
    parser.add_argument('--embedding_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0., type=float)
    parser.add_argument('--activation', default=None, type=str)
    # Evaluation Options
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_model_saved_epoch', default=None, type=int)
    parser.add_argument('--eval_visual', action='store_true')
    parser.add_argument('--visual_batch_size', default=1, type=int)
    parser.add_argument('--visual_num_images', default=10, type=int)
    # Other Options
    parser.add_argument('--save_epochs', default=10, type=int)
    parser.add_argument('--compute_baseline', action='store_true')
    return parser.parse_args()


def main(args):
    ##### Set Up Device #####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    ##### Load Dataset #####
    zipped_data_train_test = load_preprocessed_train_test_dataset(pkg_path, dataset_ver=args.dataset_ver)
    ##### Compute Baseline Error #####
    if args.compute_baseline:
        ilm(zipped_data_train_test)
    ##### Find Log Directory #####
    writername = 'epoch_'+str(args.num_epochs)+'-num_lstms_'+str(args.num_lstms)+'-hidden_size_'+str(args.hidden_size)+'-lr_'+str(args.lr)
    if args.bidirectional:
        writername = writername + '_bi'
    else:
        writername = writername + '_uni'
    writername = writername + '_zero_grad'
    if args.end_mask:
        writername = writername + '_end_mask'
    print('config: ', writername)
    logdir = join(pkg_path, 'results', 'wlstm', 'dataset_full_'+str(args.dataset_ver), writername) 
    if not args.eval:
        ##### Train ReBiL ##### 
        if isdir(logdir): # if logdir exists, delete the old one
            logdir = logdir + '_REPEATED'
            print('Need to create a REPEATED folder.')
        writer = SummaryWriter(logdir=logdir)
        train_rebil(zipped_data_train_test, writer, logdir, lr=args.lr, batch_size=args.batch_size, bidirectional=args.bidirectional, end_mask=args.end_mask, num_layers=args.num_layers, \
                    num_lstms=args.num_lstms, num_epochs=args.num_epochs, embedding_size=args.embedding_size, hidden_size=args.hidden_size, save_epochs=args.save_epochs, activation=args.activation, batch_norm=args.batch_norm, \
                    dropout=args.dropout, clip_grad_norm=args.clip_grad_norm, device=device)
        writer.close()
    else:
        ##### Evaluate ReBiL #####
        zipped_data_test = zipped_data_train_test[3:]
        model_eval = load_rebil_model(args, logdir, device=device)
        if model_eval is None:
            sys.exit(1)
        eval_rebil(zipped_data_test, model_eval, logdir, args, device=device)

if __name__ == '__main__':
    args = arg_parse()
    print()
    print()
    print('----------------------------------------------------------------------')
    print('arguments')
    print(args)
    main(args)
    print('----------------------------------------------------------------------')
    print()
    print()
