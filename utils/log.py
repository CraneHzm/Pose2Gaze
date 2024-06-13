import json
import os
import torch
import pandas as pd
import numpy as np


def save_csv_log(opt, head, value, is_create=False, file_name='results'):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    file_path = opt.ckpt + '/{}.csv'.format(file_name)
    print(file_path)
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, 'a') as f:
            df.to_csv(f, header=False, index=False)
            
            
def save_ckpt(state, opt=None, file_name = 'model.pt'):
    file_path = os.path.join(opt.ckpt, file_name)
    torch.save(state, file_path)
        

def save_options(opt):
    with open(opt.ckpt + '/options.json', 'w') as f:
        f.write(json.dumps(vars(opt), sort_keys=False, indent=4))
