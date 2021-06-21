import torch
from os.path import join, isdir, isfile
from os import listdir
import re

from src.wlstm.models import ReBiL

def load_rebil_model(args, logdir, device='cuda:0'):
    if not isdir(logdir):
        print('The folder '+logdir+' is not found.')
        return None
    if args.eval_model_saved_epoch is None:
        saved_epoch = args.num_epochs
    else:
        saved_epoch = args.eval_model_saved_epoch

    for filename in listdir(logdir):
        if isfile(join(logdir, filename)) and re.search('.*epoch_'+str(saved_epoch)+'.pt', filename):
            model_filename = join(logdir, filename)
            model = ReBiL(embedding_size=args.embedding_size, hidden_size=args.hidden_size, num_layers=args.num_layers, \
                num_lstms=args.num_lstms, bidirectional=args.bidirectional, end_mask=args.end_mask, device=device).to(device)
            checkpoint = torch.load(model_filename, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.load_lstms_dict(checkpoint['lstms_dict'])
            print(model_filename + ' is loaded.')
            return model
    print('The model is not saved at epoch '+str(saved_epoch)+' in '+logdir)
    return None
