from pathhack import pkg_path
from os.path import join, isdir
from os import mkdir
import pickle
import argparse
import time

from src.utils import load_preprocessed_train_test_dataset
from src.mif.intention_sampler import IntentionSampler
from src.mif.intention_particle import IntentionParticle
from src.mif.sample_predictor import SamplePredictor
from src.mif.intention_filter import IntentionFilter

def filtering_datasets():
    args = arg_parse()
    print()
    print()
    print('----------------------------------------------------------------------')
    print('arguments')
    print(args)
    sample_predictor = SamplePredictor(args, pkg_path, device='cuda:0')
    _, _, _, _, traj_true_test, _ = \
        load_preprocessed_train_test_dataset(pkg_path, dataset_ver=0)
    start_percent, end_percent = (args.filter_test_data-1)/10., args.filter_test_data/10.
    test_data_start_idx = int(start_percent*len(traj_true_test))
    test_data_end_idx = int(end_percent*len(traj_true_test))
    sample_true_list = traj_true_test[test_data_start_idx:test_data_end_idx]
    if args.filter_test_data == 5:
        sample_true_list = sample_true_list[:89] + sample_true_list[90:]
    print('length of test data: ', len(sample_true_list))
    infos_list = []
    if args.pred_func == 'ilm':
        pred_func = sample_predictor.ilm_pred_fit
    elif args.pred_func == 'rebil':
        pred_func = sample_predictor.rebil_pred_fit
    else:
        print('Wrong prediction function.')
        sys.exit(1)
    start_time = time.time()
    for i, sample_true in enumerate(sample_true_list):
        print('The {0}th sample in between {1}% and {2}% test data'.format(i+1, int(start_percent*100), int(end_percent*100)))
        infos = IntentionFilter(sample_true, args.particle_num_per_intent, args.num_tpp, args.tau, pred_func, \
            step_per_update=args.step_per_update, display_on=args.display_on, mutation_on=args.mutation_on)
        infos_list.append(infos)
    print('time to filter: {0:.2f} min'.format((time.time()-start_time)/60.))
    print('----------------------------------------------------------------------')
    logdir = join(pkg_path, 'results', 'mif', args.pred_func+'_tau_'+str(args.tau))
    if not args.mutation_on:
        logdir = logdir + '_no_mutate'
    print('folder: ', logdir)
    if not isdir(logdir): # if logdir exists, manually delete the old one
        mkdir(logdir)
        print('Created a folder named '+logdir)
    result_filename = 'infos_list_'+str(args.filter_test_data)+'.p'
    with open(join(logdir, result_filename), 'wb') as f:
        pickle.dump(infos_list, f)
        print(result_filename+' is created.')



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
    filtering_datasets()