import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_from_csv(path:str, test_size:float=0.1):
    data = pd.read_csv(path)
    data['LABEL'] = data['LABEL'].map(lambda x: x.lower())
    
    label_encoder = LabelEncoder()
    label_encoder.fit(data['LABEL'])
    data['LABEL_ID'] = label_encoder.transform(data['LABEL'])
    
    data.to_csv('data/sms_phishing_data_with_label.csv', index=False)
    
    return data

def split_data(data: pd.DataFrame,path=None,seed:int=None, test_size:float=0.1):
    labels = data['LABEL'].values
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, stratify=labels)
    assert list(X_train['LABEL']) == list(y_train), "WRONG SPLIT between train and test"
    assert list(X_test['LABEL']) == list(y_test), "WRONG SPLIT between train and test" 
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)
    assert list(X_train['LABEL']) == list(y_train), "WRONG SPLIT between train and valid"
    assert list(X_valid['LABEL']) == list(y_valid), "WRONG SPLIT between train and valid"
    
    X_train.to_csv(os.path.join(path,f'data_train_seed_{seed}.csv'), index=False)
    X_valid.to_csv(os.path.join(path,f'data_valid_seed_{seed}.csv'), index=False)
    X_test.to_csv(os.path.join(path,f'data_test_seed_{seed}.csv'), index=False)