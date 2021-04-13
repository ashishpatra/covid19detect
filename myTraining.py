import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

if __name__ == "__main__":
    df = pd.read_csv('covid_data.csv')
    
    feature_columns = ['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']
    
    X = df[feature_columns]
    y = df['infectionProb']
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    
    lrm = LogisticRegression()
    lrm.fit(X_train,y_train)
    
    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(lrm, file)
    file.close()
    
    