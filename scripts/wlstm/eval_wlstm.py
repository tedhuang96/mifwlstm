import torch
from src.utils import average_offset_error, max_offset_error, final_offset_error, \
    batch_iter, padding, padding_mask, load_warplstm_model


def evaluate_warplstm(
    zipped_data_test,
    logdir,
    batch_size=64,
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
    ##### Evaluate Model Performance #####
    with torch.no_grad():
        test_aoe_epoch, test_moe_epoch, test_foe_epoch = [], [], []
        test_sample_num_epoch = []
        for samples_base, samples_true, samples_loss_mask in batch_iter(traj_base_test, traj_true_test, traj_loss_mask_test, batch_size=batch_size):
            sb, sl = padding(samples_base)
            st, _ = padding(samples_true)
            sm_pred, _ = padding(samples_loss_mask) # (64, 526, 1)å
            sm_all = padding_mask(sl)
            sb, sl, st, sm_pred, sm_all = sb.to(device), sl.to(device), st.to(device), sm_pred.to(device), sm_all.to(device)
            sb_improved = model(sb, sm_pred, sl)
            test_aoe_epoch.append(average_offset_error(st, sb_improved.detach(), sm_pred).to('cpu'))
            test_moe_epoch.append(max_offset_error(st, sb_improved.detach(), sm_pred).to('cpu'))
            test_foe_epoch.append(final_offset_error(st, sb_improved.detach(), sm_pred).to('cpu'))
            test_sample_num_epoch.append(len(samples_base))
        test_aoe_epoch = torch.tensor(test_aoe_epoch)
        test_moe_epoch = torch.tensor(test_moe_epoch)
        test_foe_epoch = torch.tensor(test_foe_epoch)
        test_sample_num_epoch = torch.tensor(test_sample_num_epoch)
        test_aoe_epoch = (test_aoe_epoch * test_sample_num_epoch).sum()/test_sample_num_epoch.sum()
        test_moe_epoch = (test_moe_epoch * test_sample_num_epoch).sum()/test_sample_num_epoch.sum()
        test_foe_epoch = (test_foe_epoch * test_sample_num_epoch).sum()/test_sample_num_epoch.sum()
        print('Evaluate model: ')
        print('aoe: {0:.4f} | moe: {1:.4f} | foe: {2:.4f}\n'.format(test_aoe_epoch.item(), test_moe_epoch.item(), test_foe_epoch.item()))
