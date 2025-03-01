from logging import root
import os
import shutil
import argparse
import yaml
import pandas as pd
import numpy as np
import random
from get_data import get_data, read_params


def train_and_test(config_file):
    config = get_data(config_file)
    root_dir = config['raw_data']['data_src']
    dest = config['load_data']['preproseesd_data']
    p = config['load_data']['full_path']
    cla = config['load_data']['num_classes']

    splitr = config['train']['split_ratio']

    for k in range(cla):
        per = len(os.listdir((os.path.join(root_dir, cla[k]))))
        print(k, "->", per)
        cnt = 0
        split_ratio = round((splitr/100)*per)
        for j in os.listdir((os.path.join(root_dir,cla[k]))):
            pat = os.path.join(root_dir+'/'+cla[k])
            print(pat)
            if (cnt!=split_ratio):
                shutil.copy(pat, dest+'/'+'train'+'/'+'class_'+str(k))
                cnt+=1
            else:
                shutil.copy(pat, dest+'/'+'test'+'/'+'class_'+str(k))
        
        print("Done")


if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passed_args=args.parse_args()
    train_and_test(config_file=passed_args.config)