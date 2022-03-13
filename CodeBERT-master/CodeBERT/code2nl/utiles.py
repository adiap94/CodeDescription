import os
import random
import torch
import numpy as np
import json
import pandas as pd
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def json_dump(values, file_path=None):
    if file_path is None:
        print(json.dumps(values, sort_keys=True, indent=4, separators=(',', ': ')))
    else:
        with open(file_path, 'w') as outfile:
            json.dump(values, outfile,  sort_keys=True, indent=4, separators=(',', ': '))

def save_params(args,out_dir):
    d = vars(args)
    d.pop('device', None) # this key type is not recognised
    d["workdir"] = out_dir
    json_dump(d,os.path.join(out_dir,"config.json"))


def writeCSVLoggerFile(epoch_log,csvLoggerFile_path):
    df = pd.DataFrame([epoch_log])
    with open(csvLoggerFile_path, 'a') as f:
        df.to_csv(f, mode='a', header=f.tell() == 0, index=False)
if __name__ == "__main__":
    pass