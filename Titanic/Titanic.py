import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
#process train_data
#read data
df=pd.read_csv('./train.csv',header=0)
#print df.info()
#print df.describe()

#clean data
#1. convert string to numeric
#2. fill in missing data

#convert Sex column to numeric value
df['Sex']=df['Sex'].map({'male':0,'female':1}).astype(int)

#convert Embark column to numeric value,and fill in missing Embarked data
if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
    df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
Ports = list(enumerate(np.unique(df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int


#fill in missing Age data with median

age_median=df['Age'].dropna().median()
if len(df[df['Age'].isnull()])>0:
    df.loc[df.Age.isnull(),'Age']=age_median
    #df.Age[df.Age.isnull()]=age_median   #separate operations, slower

#remove the PassengerId,Name,Ticket,Cabin column

df=df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
#print df.head(3)

# print df.dtypes[df.dtypes.map(lambda x:x=='object')]
# print df.dtypes

#process test_data,need to do same with the test data

test_df=pd.read_csv('./test.csv',header=0)
test_df['Sex']=test_df['Sex'].map({'male':0,'female':1}).astype(int)
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
test_df.Embarked =test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int
age_median=test_df['Age'].dropna().median()
if len(test_df[test_df['Age'].isnull()])>0:
    test_df.loc[test_df.Age.isnull(),'Age']=age_median

#fill in missing data in Fare column ,assume median of their respective class

if len(test_df.Fare[test_df['Fare'].isnull()])>0:
    median_pclass=np.zeros(3)
    for i in range(3):
        median_pclass[i]=test_df.Fare[test_df.Pclass==i+1].dropna().median()
    for i in range(3):
        test_df.loc[(test_df.Pclass==i+1)&(test_df.Fare.isnull()),'Fare']=median_pclass[i]

#collect passengerId before dropping it

id=test_df['PassengerId'].values

#remove the Name,Ticket,Cabin column

test_df=test_df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

# Convert back to a numpy array
train_data=df.values
test_data=df.values
print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest.fit(train_data[:,1:],train_data[:,0])
print 'Predicting...'
output = forest.predict(test_data)

file_object=open('./prediction.csv','wb')
prediction_file=csv.writer(file_object)
prediction_file.writerow(["PassengerId","Survived"])
prediction_file.writerow(zip(id,output))
file_object.close()
print 'Done'

