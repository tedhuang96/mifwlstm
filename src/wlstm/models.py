import pickle
import torch
from torch import nn
from src.utils import average_offset_error, average_offset_error_square

class ReBiL(nn.Module):
    """
    - ReBiL is the abbreviation of Residual Bidirectional LSTM, which is the model we use for Warp LSTM.
    - ReBiL predicts the residual between the base sample
       and the true sample.
    - embed the ground truth positions directly.
    - Use the batch to train instead of single instances.
    """
    def __init__(self, embedding_size=64, hidden_size=64, num_layers=1, \
                num_lstms=3, dropout=0., bidirectional=True, end_mask=False, device='cuda:0'):
        """
        initialization.
        inputs:
            - embedding_size
            - hidden_size
            - num_layers
            - num_lstms: number of LSTMs stacked together.
            - dropout # useless by now.
            - bidirectional
            - end_mask: whether fix the start and the end of a trajectory. Useless by now.
            - device: 'cuda:0' or 'cpu'
        """
        super(ReBiL, self).__init__()
        self.embedding_size, self.hidden_size, self.num_layers, self.num_lstms = \
            embedding_size, hidden_size, num_layers, num_lstms
        if bidirectional:
            self.num_dir = 2
        else:
            self.num_dir = 1
        self.lstms = []
        for i in range(num_lstms):
            lstm = nn.LSTM(
                    input_size=embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.,
                    bidirectional=bidirectional,
                    ).to(device)
            self.lstms.append(lstm)
        self.spatial_embedding = nn.Linear(2, embedding_size)
        self.hidden2pos = nn.Linear(self.num_dir*hidden_size, 2)
        self.end_mask = end_mask
    
    def load_lstms_dict(self, lstms_dict):
        """
        load parameters of lstms. lstms is a list of lstm's, so it doesn't belong to
        the default dict saved in model.pt. Thus, it needs to be loaded separately.
        inputs:
            - lstms_dict: list of state_dict of lstms.
        """
        for i, lstm_dict in enumerate(lstms_dict):
            self.lstms[i].load_state_dict(lstm_dict)
    
    def init_hidden(self, batch, device='cuda:0'):
        """
        initialize hidden states.
        inputs:
            - batch: batch_size.
        """
        return (
            torch.zeros(self.num_layers*self.num_dir, batch, self.hidden_size).to(device),
            torch.zeros(self.num_layers*self.num_dir, batch, self.hidden_size).to(device),
        )
    
    def forward(self, sb, st, sm_pred, sm_all, sl, device='cuda:0'):
        """
        forward function.
        inputs:
            - sb: sample base. tensor. size: (batch, time_step, 2)
            - st: sample true. tensor. size: (batch, time_step, 2)
            - sm_pred: sample mask for prediction. tensor. size: (batch, time_step, 1)
            - sm_all: sample mask for observation and prediction. tensor. size: (batch, time_step, 1)
            - sl: sample length. tensor. size: (batch, ) 
        outputs:
            - loss
            - sb_improved: trajectory output which is warped sample base.
                size: (batch, time_step, 2)
        """
        sb_improved = self.inference(sb, sm_pred, sm_all, sl, device=device)
        if self.end_mask:
            loss = average_offset_error_square(st, sb_improved, sm_pred)
        else:
            loss = average_offset_error_square(st, sb_improved, sm_all)
        return loss, sb_improved
    
    def inference(self, sb, sm_pred, sm_all, sl, device='cuda:0'):
        """
        inference function.
        inputs:
            - sb: sample base. tensor. size: (batch, time_step, 2)
            - sm_pred: sample mask for prediction. tensor. size: (batch, time_step, 1)
            - sm_all: sample mask for observation and prediction. tensor. size: (batch, time_step, 1)
            - sl: sample length. tensor. size: (batch, ) 
        outputs:
            - loss
            - sb_improved: trajectory output which is warped sample base.
                size: (batch, time_step, 2)
        """
        
        batch, time_step, _ = sb.size()
        # Start and end fixed if end_mask is true.
        if self.end_mask:
            res_mask = sm_pred
        # Initialize the iteration.
        sb_improved = sb
        for lstm in self.lstms:
            sb_ebd = self.spatial_embedding(sb_improved.reshape(-1, 2))
            sb_ebd = sb_ebd.reshape(batch, time_step, self.embedding_size)
            sb_ebd = torch.nn.utils.rnn.pack_padded_sequence(sb_ebd, sl.to('cpu'), batch_first=True, enforce_sorted=False)
            hc_0 = self.init_hidden(batch, device=device)
            out, hc_t = lstm(sb_ebd, hc_0)
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            res = self.hidden2pos(out)
            res = res.reshape(batch, time_step, 2)
            if self.end_mask:
                sb_improved = sb_improved + res * res_mask
            else:
                sb_improved = sb_improved + res
        return sb_improved
    

