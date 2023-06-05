import torch
import numpy as np
import re
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def replace_entities_with_phi(df):
    PHI_PATTERN = re.compile(r'\[\*\*[^\]]+\*\*\]')
    df['assessment_clean'] = df['Assessment'].str.replace(PHI_PATTERN, '@@PHI@@')
    df['plan_clean'] = df['Plan Subsection'].str.replace(PHI_PATTERN, '@@PHI@@')
    return df 


def get_cls_repr(model, dl, device):
    '''
    extracts CLS representations from a BERT-like model 
    '''
    cls_reprs = None
    model.eval()
    for i, batch in enumerate(dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
            # hidden states from the last layer: 
            hidden_states_lst_lyr = outputs.hidden_states[-1]
            # cls repr:  32 x 768
            cls_repr = hidden_states_lst_lyr[:,0,:]
            if i == 0:
                cls_reprs = cls_repr.cpu().numpy()
            else:
                cls_reprs = np.append(cls_reprs, cls_repr.cpu().numpy(), axis=0)
    return cls_reprs