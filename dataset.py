import torch
import random
import numpy as np
from typing import Union, List, Dict
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import wordnet

class SMSDataset(Dataset):
    def __init__(self,
                 data: Dict[str, List],
                 tokenizer = None,
                 label_encoder=None,
                 max_length:int=512
                 ):

        self.data = data['TEXT']
        self.label = data['LABEL_ID']
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = label_encoder
                    
    def __len__(self):
        return len(self.label)
    
    def encode_sample(self, sample: Union[str, List[str]]):
        return self.tokenizer(sample,
                              padding=True,
                              truncation=True,
                              max_length = self.max_length,
                              return_tensors='pt')
      
    nltk.download('wordnet') 
      
    #데이터 증강 = 동의어 삽입  
    """ def augment_text(self, text: str):
        words = text.split()
        augmented_words = []

        for word in words:
            # 랜덤하게 각 단어에 대해 유의어를 찾아 삽입
            synonyms = wordnet.synsets(word)
            if len(synonyms) > 1:
                synonym = synonyms[random.randint(0, len(synonyms) - 1)].lemmas()[0].name()
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)

        return ' '.join(augmented_words) """
    
    #데이터 증강 = 랜덤 삭제
    # def random_deletion(self, text: str, p: float = 0.1):
    #     words = text.split()
    #     remaining_words = [word for word in words if random.uniform(0, 1) > p]
    #     if len(remaining_words) == 0:
    #         return text  # 삭제하지 않음
    #     else:
    #         return ' '.join(remaining_words)
    
    #데이터 증강 = 무작위 삽입
    """ def random_insertion(self, text: str, n: int = 3):
        words = text.split()
        for _ in range(n):
            new_word = "random_word"
            random_index = random.randint(0, len(words) - 1)
            words.insert(random_index, new_word)
        return ' '.join(words) """
    
    
    #데이터 증강 = 동의어 / 랜덤 삭제
    # def __getitem__(self, idx):
        
    #     text = self.data[idx]
    #     label = self.label[idx]
        
    #     #데이터 증강을 적용하는 부분
    #     #augmented_text = self.augment_text(text)
    #     augmented_text = self.random_deletion(text)

    #     elements = {'data': augmented_text,
    #                 'label': label}

    #     return elements
    
    #데이터 증강 = 무작위 삽입
    """ def __getitem__(self, idx):
        text = self.data[idx]
        label = self.label[idx]

        # 데이터 증강을 적용하는 부분
        augmented_text = self.random_insertion(text)

        elements = {'data': augmented_text,
                    'label': label}

        return elements """
    
    #기본 코드
    def __getitem__(self, idx):
            elements = {'data': self.data[idx],
                        'label': self.label[idx]}

            return elements
    
    def collate_fn(self, samples:Dict[str, List]):
        datas = [ s['data'] for s in samples]
        labels = [ s['label'] for s in samples]
        
        datas = self.encode_sample(datas)
        labels = torch.tensor(labels)

        elements = { k:v for k, v in datas.items()}
        elements['labels'] = labels

        return elements
        
    def random_seed(self, seed:int = 42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
    def id2label(self, label:Union[List[int], int]):
        return self.label_encoder.inverse_transform(label)