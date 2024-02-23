import random
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)


def load_data(path):
    return pd.read_csv(path)

def vectorize(vectorizer, data):
    vectorizer.fit(data)
    data = vectorizer.transform(data)
    
    return vectorizer, data

def train(model, X, y):
    model.fit(X, y)
    return model

def evaluate(model, X, y, average='micro'):
    predicted = model.predict(X)
    acc = accuracy_score(y_true=y, y_pred=predicted)
    f1 = f1_score(y_true=y, y_pred=predicted, average=average)
    recall = recall_score(y_true=y, y_pred=predicted, average=average)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'recall': recall
    }
    
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
def save_model(model, desc):
    with open(f"{desc}.pkl", "wb") as f:
        _model = {
            'model': model
        }
        pickle.dump(_model, f)
    
def main():
    
    train_data = load_data("./data/seed_42/data_train_seed_42.csv")
    valid_data = load_data("./data/seed_42/data_valid_seed_42.csv")
    test_data = load_data("./data/seed_42/data_test_seed_42.csv")
    # rf_means = defaultdict(list)
    # dt_means = defaultdict(list)
    nb_means = defaultdict(list)
    
    
    for seed in [0, 42, 2023]:
        print(f"----[SEED] #{seed}----")
        random_seed(seed)
        
        vectorizer = CountVectorizer()
        texts = list(train_data['TEXT'])
        vectorizer.fit(texts)
        
        X_train, y_train = vectorizer.transform(train_data['TEXT']), train_data['LABEL_ID']
        X_valid, y_valid = vectorizer.transform(valid_data['TEXT']), valid_data['LABEL_ID']
        X_test, y_test = vectorizer.transform(test_data['TEXT']), test_data['LABEL_ID']
        
        # rf = RandomForestClassifier(verbose=1)
        # dt = DecisionTreeClassifier()
        nb = ComplementNB()
        
        print("[TRAIN]")
        # rf_trained = train(rf, X_train, y_train)
        # dt_trained = train(dt, X_train, y_train)
        nb_trained = train(nb, X_train, y_train)
        
        print("[EVAL]")
        # rf_result = evaluate(rf_trained, X_test, y_test, average='macro')
        # dt_result = evaluate(dt_trained, X_test, y_test, average='macro')
        nb_result = evaluate(nb_trained, X_test, y_test, average='macro')
        
        # for k1, k2 in zip(rf_result, dt_result):
        for k3 in nb_result:
            # rf_means[k1].append(rf_result[k1])
            # dt_means[k2].append(dt_result[k2])
            nb_means[k3].append(nb_result[k3])
        
        # save_model(rf_trained, f"RandomForest_seed{seed}")
        # save_model(dt_trained, f"DecisionTree_seed{seed}")
    
    print("="*64)
    # print("Random Forest results:", {k: np.mean(v) for k, v in rf_means.items()})
    # print("Decision Tree results:",{k: np.mean(v) for k, v in dt_means.items()})
    print("Naive Bayes results:",{k: np.mean(v) for k, v in nb_means.items()})
    
if __name__ == '__main__':
    main()