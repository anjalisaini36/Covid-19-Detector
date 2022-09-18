import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data, ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data) * ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[train_indices]
    
if __name__ =="__main__":

    # Read the data
    df=pd.read_csv('data1.csv')
    train,test=data_split(df, 0.2)
    X_train=train[['Fever','Bodypain','Age','RunnyNose','DiffBreath']].to_numpy()
    X_test=test[['Fever','Bodypain','Age','RunnyNose','DiffBreath']].to_numpy()
    Y_train=train[['InfectionProb']].to_numpy().reshape(2400)
    Y_test=test[['InfectionProb']].to_numpy().reshape(2400)
    clf= LogisticRegression()
    clf.fit(X_train, Y_train)

    # Code for inference
    inputFeatures=[100, 1, 22, -1, 1]
    infProb=clf.predict_proba([inputFeatures])[0][1]


filename = 'model.pkl'
file = open(filename,'wb')


pickle.dump(clf,file)
file.close()

