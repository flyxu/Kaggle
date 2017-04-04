import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import  cross_val_score

def cross():
    # loading data
    print 'Loading training data...'
    data=pd.read_csv('./digittrain.csv',header=0)
    #print data.dtypes
    x_train=data.values[:,1:]#delete header
    y_train=data.values[:,0]
    print 'starting learning...'
    n_trees=[10,15,20,25,30,40,50,70,100,150]
    scores_mean=[]
    scores_std=[]
    for n_tree in n_trees:
        cf=RandomForestClassifier(n_tree)
        score=cross_val_score(cf,x_train,y_train)
        scores_mean.append(score.mean())
        scores_std.append(score.std())
    mean_array=np.array(scores_mean)
    std_array=np.array(scores_std)
    print 'Score:',mean_array
    print 'Std',std_array
    xy = [10, 150, 0.9, 1]
    plt.axis(xy)
    plt.plot(n_trees,scores_mean)
    plt.plot(n_trees,std_array+std_array,'b--')
    plt.plot(n_trees,std_array-std_array,'b--')
    plt.ylabel('CV score')
    plt.xlabel('trees')
    plt.savefig('cv_trees.png')
    plt.show()


if __name__=="__main__":
    cross()

