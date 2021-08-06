from os.path import join, abspath
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.wlstm.utils import load_rebil_model

class SamplePredictor():
    def __init__(self, args, pkg_path, device='cuda:0'):
        self.device = device
        self.rebil_stages = self.load_rebil_stages(args, pkg_path)
        

    def load_rebil_stages(self, args, pkg_path):
        # ReBiL is the abbreviation of Residual Bidirectional LSTM, which is the model we use for Warp LSTM.
        # need to use --bidirectional as argument
        writername = 'epoch_'+str(args.num_epochs)+'-num_lstms_'+str(args.num_lstms)+'-hidden_size_'+str(args.hidden_size)+'-lr_'+str(args.lr)
        if args.bidirectional:
            writername = writername + '_bi'
        else:
            writername = writername + '_uni'
        writername = writername + '_zero_grad'
        if args.end_mask:
            writername = writername + '_end_mask'
        print('config: ', writername)
        rebil_stages = []
        for dataset_ver in [0, 25, 50, 75]:
            logdir = join(pkg_path, 'results', 'wlstm', 'dataset_full_'+str(args.dataset_ver), writername)
            rebil = load_rebil_model(args, logdir, device=self.device)
            rebil_stages.append(rebil)
        return rebil_stages

    def ilm_pred_fit(self, x_obs, sample_goals, intentions, intention_coordinates, \
        intention_num=3, num_tpp=12, silent=True):
        """
        inputs:
            - x_obs: np. (t, 2)

        outputs:
            - sample_pred: np. (340, 20, 2)
            - sample_full_pred_arr: np array of objects. (340, )
                - sample: np. (variable_t, 2)
        """
        sample_full_pred_arr = np.empty(len(intentions)).astype(object)
        sample_pred = np.empty((len(intentions), num_tpp, 2))
        heuristic_dist = np.linalg.norm(intention_coordinates-x_obs[-1], axis=1)
        mean_vel_mag = np.mean(np.linalg.norm(x_obs[1:]-x_obs[:-1], axis=1))
        step_num_noise = np.random.randint(2, 10, size=intention_num)
        heuristic_num_steps = (heuristic_dist/mean_vel_mag).astype(int)+step_num_noise
        heuristic_num_steps[heuristic_num_steps<num_tpp+1] = num_tpp+1
        for i in range(intention_num):
            sample_goals_i = sample_goals[intentions==i]
            sample_base_pred_i = (np.linspace(x_obs[-1], sample_goals_i, num=heuristic_num_steps[i])).transpose(1,0,2) # (batch, time_step, 2)
            x_obs_copy = x_obs[np.newaxis, :-1] * np.ones((sample_base_pred_i.shape[0],1,1))
            sample_base_i = np.concatenate((x_obs_copy, sample_base_pred_i), axis=1)
            sb = sample_base_i
            sample_full_pred_arr[np.where(intentions==i)[0]]=list(sb[:, -heuristic_num_steps[i]:])
            sample_pred[intentions==i] = sb[:, -heuristic_num_steps[i]:-heuristic_num_steps[i]+num_tpp]
        return sample_pred, sample_full_pred_arr

    
    def rebil_pred_fit(self, x_obs, sample_goals, intentions, intention_coordinates, intention_num=3, num_tpp=12, silent=True):
        """
        inputs:
            - x_obs: np. (t, 2)

        outputs:
            - sample_pred: np. (340, 20, 2)
            - sample_full_pred_arr: np array of objects. (340, )
                - sample: np. (variable_t, 2)
        """
        sample_full_pred_arr = np.empty(len(intentions)).astype(object)
        sample_pred = np.empty((len(intentions), num_tpp, 2))
        heuristic_dist = np.linalg.norm(intention_coordinates-x_obs[-1], axis=1)
        mean_vel_mag = np.mean(np.linalg.norm(x_obs[1:]-x_obs[:-1], axis=1))
        step_num_noise = np.random.randint(2, 10, size=intention_num)
        heuristic_num_steps = (heuristic_dist/mean_vel_mag).astype(int)+step_num_noise
        heuristic_num_steps[heuristic_num_steps<num_tpp+1] = num_tpp+1
        for i in range(intention_num):
            sample_goals_i = sample_goals[intentions==i]
            sample_base_pred_i = (np.linspace(x_obs[-1], sample_goals_i, num=heuristic_num_steps[i])).transpose(1,0,2) # (batch, time_step, 2)
            x_obs_copy = x_obs[np.newaxis, :-1] * np.ones((sample_base_pred_i.shape[0],1,1))
            if len(self.rebil_stages) == 1:
                rlstm_index = 0
            else:
                obs_ratio = float(len(x_obs)) / (heuristic_num_steps[i]+len(x_obs))
                if obs_ratio <= 0.125:
                    rlstm_index = 0
                elif obs_ratio <= 0.375:
                    rlstm_index = 1
                elif obs_ratio <= 0.625:
                    rlstm_index = 2
                else:
                    rlstm_index = 3
            res_lstm_eval = self.rebil_stages[rlstm_index]
            test_traj_full = self.rebil_inference(x_obs_copy, sample_base_pred_i, res_lstm_eval)
            if not silent:
                for j in range(test_traj_full.shape[0]):
                    plt.plot(test_traj_full[j,:,0],test_traj_full[j,:,1])
                print(test_traj_full.shape)
            test_traj_pred = test_traj_full[:, -heuristic_num_steps[i]:] # (b, t, 2)
            test_traj_pred = test_traj_pred - test_traj_pred[:,0:1]+x_obs[-1] # (b,t,2)
            test_traj_pred = test_traj_pred[:, 1:] # (b, t-1, 2)
            sample_full_pred_arr[np.where(intentions==i)[0]]=list(test_traj_pred)
            sample_pred[intentions==i] = test_traj_pred[:,:num_tpp]
            
        return sample_pred, sample_full_pred_arr


    def rebil_inference(self, x_obs_copy, sample_base_pred, model_eval):
        """
        inputs:
            - x_obs_copy: np # (100, 10, 2)
            - sample_base_pred: np # (100, 64, 2)
        outputs:
            # - print aoe and moe on test dataset by using the intention-aware linear model.
        """
        sb_np = np.concatenate((x_obs_copy, sample_base_pred), axis=1)
        if sb_np.shape[0] == 0:
            return sb_np
        batch_size, obs_time_step = x_obs_copy.shape[0], x_obs_copy.shape[1]
        pred_time_step = sample_base_pred.shape[1]
        loss_mask_zero = np.zeros((batch_size, obs_time_step, 1))
        loss_mask_one = np.ones((batch_size, pred_time_step, 1))
        loss_mask = np.concatenate((loss_mask_zero, loss_mask_one), axis=1)
        sb = torch.tensor(sb_np).float()
        sl = (torch.ones(batch_size)*(obs_time_step+pred_time_step)).float()
        sm_pred = torch.tensor(loss_mask).float()
        sm_all = torch.ones(batch_size, obs_time_step+pred_time_step, 1).float()
        sb_improved = model_eval.inference(sb.to(self.device), sm_pred.to(self.device), sm_all.to(self.device), sl.to(self.device))
        test_traj_full = np.copy(sb_improved.to('cpu').detach().numpy())
        return test_traj_full
