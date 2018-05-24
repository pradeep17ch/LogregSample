import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder  
import csv

pd.set_option('mode.chained_assignment', None)

train = pd.read_csv('./dataset/train.csv')
test = pd.read_csv('./dataset/test.csv')

print(train)
print(test)

train.info()
test.info()

le = LabelEncoder()

for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']:
	if type(train[col][0]) is str:
		train_dropped = train[col].dropna()
		train_trans = le.fit_transform(train_dropped)
		k=0
		for i,x in enumerate(train[col]):
			if type(x) is str:
				train[col][i] = train_trans[k]
				k += 1
	
	median_t = np.median(train_trans)
	for i,x in enumerate(train[col]):
		if np.isnan(x):
			train[col][i] = median_t

	if type(test[col][0]) is str:
		test_dropped = test[col].dropna()
		test_trans = le.transform(test_dropped)
		k=0
		for i,x in enumerate(test[col]):
			if type(x) is str:
				test[col][i] = test_trans[k]
				k += 1
	
	median_t = np.median(train_trans)
	for i,x in enumerate(test[col]):
		if np.isnan(x):
			test[col][i] = median_t

print(train)
print(test)

X_train = train.drop('P', axis=1)
Y_train = train['P']
X_test = test.copy()

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

print(Y_pred)

with open('./dataset/submission.csv', 'wb') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['id', 'P'])
    for i,x in enumerate(Y_pred):
    	temp = []
    	temp.append(test['id'][i])
    	temp.append(Y_pred[i])
    	filewriter.writerow(temp)
