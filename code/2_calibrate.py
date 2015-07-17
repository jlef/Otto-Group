# -*- coding: utf-8 -*-
"""
Created on Mon May 04 12:31:26 2015

@author: jlef

One vs All isotonic calibration of GBM predictions
"""
import os
##CHANGE THE PATH
os.chdir(r'...Otto-Group\\code')

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression

# import data
train = pd.read_csv(r'..\\input\\train.csv')

cv10fold = pd.read_csv(r'csvs\\10foldCV_xB.csv')
cv10fold['target'] = train.target.values
cv10fold = cv10fold.reindex(np.random.permutation(cv10fold.index))
id = cv10fold.id.values
cv10fold = cv10fold.drop('id', axis=1)

target = cv10fold.target.values
y_train = pd.get_dummies(target)
cv10fold = cv10fold.drop('target', axis=1)

test = pd.read_csv(r'csvs\\submit_xB.csv')
testId = test.id.values
test = test.drop('id', axis=1)

##one vs all
ir1 = IsotonicRegression()
ir2 = IsotonicRegression()
ir3 = IsotonicRegression()
ir4 = IsotonicRegression()
ir5 = IsotonicRegression()
ir6 = IsotonicRegression()
ir7 = IsotonicRegression()
ir8 = IsotonicRegression()
ir9 = IsotonicRegression()

y_1 = ir1.fit_transform(cv10fold.ix[:,0], y_train.ix[:,0])
y_2 = ir2.fit_transform(cv10fold.ix[:,1], y_train.ix[:,1])
y_3 = ir3.fit_transform(cv10fold.ix[:,2], y_train.ix[:,2])
y_4 = ir4.fit_transform(cv10fold.ix[:,3], y_train.ix[:,3])
y_5 = ir5.fit_transform(cv10fold.ix[:,4], y_train.ix[:,4])
y_6 = ir6.fit_transform(cv10fold.ix[:,5], y_train.ix[:,5])
y_7 = ir7.fit_transform(cv10fold.ix[:,6], y_train.ix[:,6])
y_8 = ir8.fit_transform(cv10fold.ix[:,7], y_train.ix[:,7])
y_9 = ir9.fit_transform(cv10fold.ix[:,8], y_train.ix[:,8])

#container
cv10fold.calibrated = pd.DataFrame({'id' : id
                , 'Class_1' : y_1
                , 'Class_2' : y_2
                , 'Class_3' : y_3
                , 'Class_4' : y_4
                , 'Class_5' : y_5
                , 'Class_6' : y_6
                , 'Class_7' : y_7
                , 'Class_8' : y_8
                , 'Class_9' : y_9
                
                })
                
cols = cv10fold.calibrated.columns.tolist()
cols = cols[-1:] + cols[:-1]
cv10fold.calibrated = cv10fold.calibrated[cols]

#for validation purposes
cv10fold.calibrated.to_csv('csvs\\cv10fold.calibrated.csv', index=False)

yt_1 = ir1.predict(test.ix[:,0])
yt_2 = ir2.predict(test.ix[:,1])
yt_3 = ir3.predict(test.ix[:,2])
yt_4 = ir4.predict(test.ix[:,3])
yt_5 = ir5.predict(test.ix[:,4])
yt_6 = ir6.predict(test.ix[:,5])
yt_7 = ir7.predict(test.ix[:,6])
yt_8 = ir8.predict(test.ix[:,7])
yt_9 = ir9.predict(test.ix[:,8])

test.calibrated = pd.DataFrame({'id' : testId
                    , 'Class_1' : yt_1
                    , 'Class_2' : yt_2
                    , 'Class_3' : yt_3
                    , 'Class_4' : yt_4
                    , 'Class_5' : yt_5
                    , 'Class_6' : yt_6
                    , 'Class_7' : yt_7
                    , 'Class_8' : yt_8
                    , 'Class_9' : yt_9
                })

cols = test.calibrated.columns.tolist()
cols = cols[-1:] + cols[:-1]
test.calibrated = test.calibrated[cols]

for x in range(1, 10):
    incompleteCases = pd.isnull(test.calibrated.ix[:,x])
    test.calibrated.ix[incompleteCases,x] = 1 - test.calibrated.ix[incompleteCases,[-0,-x]].sum(axis=1)
    
test.calibrated[test.calibrated < 0] = 0

#Add the calibrated predictions with the original predictions
test.calibrated.ix[:,1:10] = test+test.calibrated.ix[:,1:10]

test.calibrated.to_csv('submission.calibrated.csv', index=False)

