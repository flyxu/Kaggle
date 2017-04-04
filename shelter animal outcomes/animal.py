import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import sys
from datetime import datetime

def munge(data,train):
    data['HasName']=data['Name'].fillna(0)
    data.loc[data['HasName']!=0,'HasName']=1
    #print data.dtypes
    data['HasName']=data['HasName'].astype(int)
    data['AnimalType']=data['AnimalType'].map({'Cat':0,'Dog':1})
    #print data.dtypes
    gender={'Neutered Male':1,'Spayed Female':2,'Intact Male':3,'Intact Female':4,'Unknown':5,np.nan:0}
    data['SexuponOutcome']=data['SexuponOutcome'].map(gender)
    #print data.dtypes
    if(train):
        data.drop(['AnimalID','OutcomeSubtype'],axis=1,inplace=True)
        data['OutcomeType']=data['OutcomeType'].map({'Return_to_owner':4,'Euthanasia':3,'Adoption':0,'Transfer':5,'Died':2})
    #print data.dtypes
    def agetodays(x):
        try:
            y=x.split()
        except:
            return None
        if 'year' in y[1]:
            return float(y[0])*365
        elif 'month' in y[1]:
            return float(y[0])*(365/12)
        elif 'week' in y[1]:
            return float(y[0])*7
        elif 'day' in y[1]:
            return float(y[0])
    data['AgeInDays']=data['AgeuponOutcome'].map(agetodays)
    data.loc[data['AgeInDays'].isnull(),'AgeInDays']=data['AgeInDays'].median()
    data['Year'] = data['DateTime'].str[:4].astype(int)
    data['Month'] = data['DateTime'].str[5:7].astype(int)
    data['Day'] = data['DateTime'].str[8:10].astype(int)
    data['Hour'] = data['DateTime'].str[11:13].astype(int)
    data['Minute'] = data['DateTime'].str[14:16].astype(int)
    data['Name+Gender']=data['HasName']+data['SexuponOutcome']
    data['Type+Gender']=data['AnimalType']+data['SexuponOutcome']
    def mix(x):
        if 'Mix' in x:
            return 1
        else:
            return 0
    data['IsMix']=data['Breed'].map(mix)
    #data['IsMix'] = data['Breed'].str.contains('mix', case=False).astype(int)
    data=data.drop(['AgeuponOutcome', 'Name', 'Breed', 'Color', 'DateTime'],axis=1)
    return data

def bestparam(data):
    rfc=RandomForestClassifier()
    param_grid={'n_estimators':[50,400],'max_features':['auto','sqrt','log2']}
    cv_rfc=GridSearchCV(estimator=rfc,param_grid=param_grid,cv=5)
    cv_rfc.fit(data[0:,1:],data[0:,0])
    return cv_rfc.best_params_


if __name__=='__main__':
    train=pd.read_csv('./train.csv')
    test=pd.read_csv('./test.csv')
    pd_train=munge(train,True)
    pd_test=munge(test,False)
    pd_test.drop('ID',inplace=True,axis=1)
    print 'pd_train',pd_train.iloc[0:3,1:]
    train=pd_train.values
    test=pd_test.values
    #print train.shape
    #print test.shape
    #print train[0:5]
    #best=bestparam(train)
    #print best
    print 'Predicting...\n'
    forest=RandomForestClassifier(n_estimators=400,max_features='auto')
    forest=forest.fit(train[0:,1:],train[0:,0])
    predictions=forest.predict_proba(test)
    output = pd.DataFrame(predictions, columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
    id=range(1,len(output)+1)
    output.insert(0,'ID',id)
    print output.iloc[0:3]
    output.to_csv('predictions.csv',index=False)
    print 'Done.'