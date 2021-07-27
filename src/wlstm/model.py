import torch
from torch import nn

class WarpLSTM(nn.Module):
    r"""Warp the nominal trajectory by adding the residual to approach the ground truth trajectory."""
    
    def __init__(
        self,
        embedding_size=64,
        hidden_size=64,
        num_layers=1,
        num_lstms=3,
        dropout=0.,
        bidirectional=True,
        end_mask=False,
        ):
        r"""
        Initialize the components.
        inputs:
            - embedding_size # embedding dimension.
            - hidden_size # dimension of hidden state.
            - num_layers # number of layers of one LSTM.
            - num_lstms # number of LSTMs.
            - dropout # LSTM dropout.
            - bidirectional # Whether LSTM is bidirectional or unidirectional.
            - end_mask # whether observation and goal position of the trajectory are masked.
        Outputs:
            - None
        """
        super(WarpLSTM, self).__init__()
        self.embedding_size, self.end_mask = embedding_size, end_mask
        self.spatial_embedding = nn.Linear(2, self.embedding_size)
        self.lstms = nn.ModuleList([
            nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            ) for _ in range(num_lstms)
        ])
        if bidirectional:
            self.hidden2pos = nn.Linear(2*hidden_size, 2)
        else:
            self.hidden2pos = nn.Linear(hidden_size, 2)
    
    def forward(self, sb, sm_pred, sl, device='cuda:0'):
        r"""
        Forward function.
        inputs:
            - sb 
                # sample base, i.e. nominal prediction.
                # (batch, time_step, 2)
            - sm_pred
                # sample mask for prediction.
                # (batch, time_step, 1)
            - sl
                # sample length.
                # (batch, ) 
        outputs:
            - sb_improved
                # trajectory output which is warped sample base.
                size: (batch, time_step, 2)
        """
        batch, time_step, _ = sb.size()
        sb_improved = sb
        for lstm in self.lstms:
            sb_ebd = self.spatial_embedding(sb_improved.reshape(-1, 2))
            sb_ebd = sb_ebd.reshape(batch, time_step, self.embedding_size)
            sb_ebd = torch.nn.utils.rnn.pack_padded_sequence(sb_ebd, sl.to('cpu'), batch_first=True, enforce_sorted=False)
            out, _ = lstm(sb_ebd)
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            res = self.hidden2pos(out)
            res = res.reshape(batch, time_step, 2)
            if self.end_mask:
                # observation and goal position are fixed if end_mask is true.
                sb_improved = sb_improved + res * sm_pred
            else:
                sb_improved = sb_improved + res
        return sb_improved