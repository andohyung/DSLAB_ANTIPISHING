import os

################## GPU #####################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
############################################

import torch
import numpy as np
import random
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.text_utils import load_from_csv, split_data
from utils.data_utils import make_loader
from dataset import SMSDataset
from trainer import Trainer
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast  # Changed imports for RoBERTa

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batchsize', type=int, default=8)
    args = parser.parse_args()
    
    seed = args.seed
    random_seed(seed=seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-----CUDA DEVICES-----")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    print("-" * 22)
    
    id2label = {0: 'ham',
                1: 'smishing',
                2: 'spam'}

    label2id = {'ham': 0,
                'smishing': 1,
                'spam': 2}
    
    model_path = 'roberta-base'  # Use RoBERTa model instead of BERT
    output_dir = f"outputs/roberta_epochs_5_seed_{seed}_lr_3e-05_batch_{args.batchsize}"
    
    roberta_model = RobertaForSequenceClassification.from_pretrained(os.path.join(output_dir, 'model_best_valid_accuracy'))
    roberta_tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    
    test_data = pd.read_csv(f"data/seed_{seed}/data_test_seed_{seed}.csv")
    test_set = SMSDataset(data=test_data,
                         tokenizer=roberta_tokenizer,
                         max_length=roberta_model.config.max_position_embeddings)  # Adjusted for RoBERTa
    test_loader = make_loader(dataset=test_set, batch_size=args.batchsize, seed=seed)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    trainer = Trainer(test_loader=test_loader,
                      device=device,
                      outputs_dir=output_dir)
    
    test_output = trainer.test(model=roberta_model)  # Changed from bert_model to roberta_model
    print(f"test_output:{test_output}")
