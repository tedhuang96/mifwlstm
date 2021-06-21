from pathhack import pkg_path
from os.path import join
import pickle
import numpy as np
import argparse
import time

def is_neighbor(correct_intention, top_intentions, half_range=1):
    """
    inputs:
    ----- correct_intention: int.
    ----- top_intentions: np.
    ----- half_range: the distance between the furthest acceptable neighbor and the correct intention.
    outputs:
    ----- neighbor_flag: bool.
    """
    neighbors_correct_intention = np.arange(-half_range, half_range+1)+correct_intention # e.g. (-2, -1, 0, 1, 2)
    for i in range(len(neighbors_correct_intention)):
        if neighbors_correct_intention[i] < 0:
            neighbors_correct_intention[i] += 34
        elif neighbors_correct_intention[i] > 33:
            neighbors_correct_intention[i] -= 34
    for neighbor in neighbors_correct_intention:
        if neighbor in top_intentions:
            return True
    return False

def intention_accuracy_check():
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
    mean_class_acc_list = []
    final_class_acc_list = []
    start_time = time.time()
    for i, infos in enumerate(infos_list_all):
        [correct_intention, percentage_hist, intention_dist_hist, particle_intention_hist, long_term_pred_hist, long_term_true_hist,\
                long_term_obs_hist] = infos
        intention_classification_list = []
        for ts in range(len(long_term_obs_hist)):
            intention_dist = intention_dist_hist[ts]
            # top 3 intentions # use top 3 or top 1 trajectory
            top_intentions = np.flip(np.argsort(intention_dist))[:args.num_top_intentions] # ndarray
            intention_classification_list.append(is_neighbor(correct_intention, top_intentions, half_range=1))
        mean_class_acc_list.append(np.mean(intention_classification_list))
        final_class_acc_list.append(intention_classification_list[-1])
    mean_class_acc = np.mean(mean_class_acc_list)
    final_class_acc = np.mean(final_class_acc_list)
    print('number of top intentions: {0}'.format(args.num_top_intentions))
    print('mean_class_acc:{0:.3f}, final_class_acc:{1:.3f}'.format(mean_class_acc, final_class_acc))


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
    intention_accuracy_check() # evaluate the filter test.