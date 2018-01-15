import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True) #not dropping this column brings down accuracy to 56%, CLEAN THE DATA

X = np.array(df.drop(['class'],1)) #features = everything but class
y = np.array(df['class']) #labels = class (yes/no, malignate/begnin, on/off, basically the answer)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1]])

example_measures = example_measures.reshape(len(example_measures),-1) #(sample size=2,)

prediction = clf.predict(example_measures)

print(prediction)
