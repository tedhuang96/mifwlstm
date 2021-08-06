from pathhack import pkg_path
from os.path import isdir, join
import time

import torch
import argparse
import numpy as np
from tensorboardX import SummaryWriter

from src.utils import average_offset_error, batch_iter, \
    padding, padding_mask, load_preprocessed_train_test_dataset, ilm, average_offset_error_square
from src.wlstm.model import WarpLSTM

def train_warplstm(
    zipped_data_train_test,
    writer,
    logdir,
    lr=1e-4,
    batch_size=64,
    bidirectional=True,
    end_mask=False,
    num_layers=1,
    num_lstms=3,
    num_epochs=200,
    embedding_size=128,
    hidden_size=128,
    save_epochs=50,
    dropout=0.,
    clip_grad_norm=10.,
    device='cuda:0',
):
    ##### Unzip Data #####
    traj_base_train, traj_true_train, traj_loss_mask_train, \
        traj_base_test, traj_true_test, traj_loss_mask_test = zipped_data_train_test
    ##### Initialize Model & Optimizer #####
    model = WarpLSTM(
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_lstms=num_lstms,
        dropout=dropout,
        bidirectional=bidirectional,
        end_mask=end_mask,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print()
    print('MODEL INITIALIZATION')
    print('learning rate: ', lr)
    print('model architecture: ')
    print(model)

    ##### Training #####
    print()
    print('TRAINING STARTED')
    print('# EPOCHS: ', num_epochs)
    train_loss_task, test_loss_task, train_aoe_task, test_aoe_task = [],[],[],[]
    for epoch in range(1, num_epochs+1):
        ## train ##
        epoch_start_time = time.time()
        model.train()
        train_sample_num_epoch = []
        train_loss_epoch = []
        train_results_epoch = []
        for samples_base, samples_true, samples_loss_mask in batch_iter(traj_base_train, traj_true_train, traj_loss_mask_train, batch_size=batch_size):
            optimizer.zero_grad()
            sb, sl = padding(samples_base)
            st, _ = padding(samples_true)
            sm_pred, _ = padding(samples_loss_mask) # (64, 526, 1)
            sm_all = padding_mask(sl)
            sb, sl, st, sm_pred, sm_all = sb.to(device), sl.to(device), st.to(device), sm_pred.to(device), sm_all.to(device)
            sb_improved = model(sb, sm_pred, sl)
            if end_mask:
                loss = average_offset_error_square(st, sb_improved, sm_pred)
            else:
                loss = average_offset_error_square(st, sb_improved, sm_all)
            train_results_epoch.append(average_offset_error(st, sb_improved.detach(), sm_pred).to('cpu'))
            train_sample_num_epoch.append(len(samples_base))
            train_loss_epoch.append(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
        training_epoch_period = time.time() - epoch_start_time
        training_epoch_period_per_sample = training_epoch_period/len(traj_base_train)
        ## eval ##
        with torch.no_grad():
            model.eval()
            test_loss_epoch = []
            test_results_epoch = []
            test_sample_num_epoch = []
            for samples_base, samples_true, samples_loss_mask in batch_iter(traj_base_test, traj_true_test, traj_loss_mask_test, batch_size=batch_size):
                sb, sl = padding(samples_base)
                st, _ = padding(samples_true)
                sm_pred, _ = padding(samples_loss_mask) # (64, 526, 1)
                sm_all = padding_mask(sl)
                sb, sl, st, sm_pred, sm_all = sb.to(device), sl.to(device), st.to(device), sm_pred.to(device), sm_all.to(device)
                sb_improved = model(sb, sm_pred, sl)
                if end_mask:
                    loss = average_offset_error_square(st, sb_improved, sm_pred)
                else:
                    loss = average_offset_error_square(st, sb_improved, sm_all)
                test_results_epoch.append(average_offset_error(st, sb_improved.detach(), sm_pred).to('cpu'))
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
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss_task,
                    'val_loss': test_loss_task,
                    'train_aoe': train_aoe_task,
                    'val_aoe': test_aoe_task,      
                    }, model_filename)
                print(model_filename+' is saved.')
    print()


def arg_parse():
    parser = argparse.ArgumentParser()
    # Dataset Options
    parser.add_argument('--dataset_ver', default=0, type=int)
    # Optimization Options
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--clip_grad_norm', default=10., type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    # Model Options
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--end_mask', action='store_true')
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--num_lstms', default=3, type=int)
    parser.add_argument('--embedding_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0., type=float)
    # Other Options
    parser.add_argument('--save_epochs', default=50, type=int)
    parser.add_argument('--compute_baseline', action='store_true')
    parser.add_argument('--random_seed', default=0, type=int)
    return parser.parse_args()


def main(args):
    ##### Set Up Device #####
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    ##### Load Dataset #####
    zipped_data_train_test = load_preprocessed_train_test_dataset(pkg_path, dataset_ver=args.dataset_ver)
    ##### Compute Baseline Error #####
    if args.compute_baseline:
        ilm(zipped_data_train_test)
    ##### Find Log Directory #####
    writername = 'epoch_'+str(args.num_epochs)+'-num_lstms_'+str(args.num_lstms)+'-lr_'+str(args.lr)
    if args.bidirectional:
        writername = writername + '_bi'
    else:
        writername = writername + '_uni'
    if args.end_mask:
        writername = writername + '_end_mask'
    print('config: ', writername)
    logdir = join(pkg_path, 'results', 'wlstm', 'dataset_full_'+str(args.dataset_ver), writername) 
    
    ##### Train ReBiL ##### 
    if isdir(logdir):
        raise RuntimeError('The result directory was already created and used.')
    writer = SummaryWriter(logdir=logdir)
    train_warplstm(
        zipped_data_train_test,
        writer,
        logdir,
        lr=args.lr,
        batch_size=args.batch_size,
        bidirectional=args.bidirectional,
        end_mask=args.end_mask,
        num_layers=args.num_layers,
        num_lstms=args.num_lstms,
        num_epochs=args.num_epochs,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        save_epochs=args.save_epochs,
        dropout=args.dropout,
        clip_grad_norm=args.clip_grad_norm,
        device=device)
    writer.close()
    

if __name__ == '__main__':
    args = arg_parse()
    print('\n\n----------------------------------------------------------------------')
    print('arguments')
    print(args)
    main(args)
    print('----------------------------------------------------------------------\n\n')
