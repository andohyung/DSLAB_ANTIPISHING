import os
import torch
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from utils.text_utils import load_from_csv, split_data
import argparse

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    seed=args.seed
    random_seed(seed=seed)
    os.makedirs(f'data/seed_{seed}')
    
    data_path = 'data/sms_phishing_data_with_label.csv'
    data = pd.read_csv(data_path)
    
    split_data(data=data,
               path=f'data/seed_{seed}',
               seed=seed,
               test_size=0.1)