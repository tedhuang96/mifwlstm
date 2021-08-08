from pathhack import pkg_path
from os.path import isdir, join

import torch
import argparse
import numpy as np
from tensorboardX import SummaryWriter

from src.utils import load_preprocessed_train_test_dataset, ilm, args_to_logdir
from scripts.wlstm.train_wlstm import train_warplstm
from scripts.wlstm.eval_wlstm import evaluate_warplstm
from scripts.wlstm.visual_wlstm import visualize_warplstm

def arg_parse():
    parser = argparse.ArgumentParser()
    # General Options
    parser.add_argument('--dataset_ver', default=0, type=int)
    parser.add_argument('--mode', default='train', help='train, eval, or visual.')
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
    # Evaluation Options
    parser.add_argument('--saved_model_epoch', default=None, type=int)
    parser.add_argument('--visual_batch_size', default=1, type=int)
    parser.add_argument('--visual_num_images', default=10, type=int)
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
    logdir = args_to_logdir(args, pkg_path)
    
    if args.mode == 'train':
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
            device=device,
        )
        writer.close()
    elif args.mode == 'eval':
        if not isdir(logdir):
            raise RuntimeError('The folder '+logdir+' is not found.')
        zipped_data_test = zipped_data_train_test[3:]
        evaluate_warplstm(
            zipped_data_test,
            logdir,
            batch_size=args.batch_size,
            bidirectional=args.bidirectional,
            end_mask=args.end_mask,
            num_layers=args.num_layers,
            num_lstms=args.num_lstms,
            num_epochs=args.num_epochs,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            saved_model_epoch=args.saved_model_epoch,
            device=device,
        )
    elif args.mode == 'visual':
        if not isdir(logdir):
            raise RuntimeError('The folder '+logdir+' is not found.')
        zipped_data_test = zipped_data_train_test[3:]
        visualize_warplstm(
            zipped_data_test,
            logdir,
            visual_batch_size=args.visual_batch_size,
            visual_num_images=args.visual_num_images,
            bidirectional=args.bidirectional,
            end_mask=args.end_mask,
            num_layers=args.num_layers,
            num_lstms=args.num_lstms,
            num_epochs=args.num_epochs,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            saved_model_epoch=args.saved_model_epoch,
            device=device,
        )

if __name__ == '__main__':
    print('\n\n----------------------------------------------------------------------')
    print('arguments')
    args = arg_parse()
    print(args)
    main(args)
    print('----------------------------------------------------------------------\n\n')
