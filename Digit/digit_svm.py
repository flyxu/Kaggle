import numpy
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pandas as pd

print 'reading training data...'
data=pd.read_csv('./digittrain.csv')
x_train=data.values[:,1:]
y_train=data.values[:,0]
print 'Reduction...'
pca=PCA(n_components=35,whiten=True)
pca.fit(x_train)
x_train=pca.transform(x_train)
print x_train[0:3]
print 'svm training...'
svc=SVC()
svc.fit(x_train,y_train)

print 'reading testing data...'
test_data=pd.read_csv('./digittest.csv')
x_test=test_data.values
x_test=pca.transform(x_test)

print 'Predicting...'
predict=svc.predict(x_test)

print 'Saving...'
file=open('./predict.csv','w')
file.write('"ImageId","Label"\n')
count=0
for p in predict:
    count+=1
    file.write(str(count)+','+str(p)+'\n')
