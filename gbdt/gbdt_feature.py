#coding:utf-8

import xgboost as xgb
from sklearn import cross_validation
import load_data
import cPickle as pickle
import csv
from collections import Counter
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

train_x = pickle.load(open("../data/train_x", "rb"))
train_y = pickle.load(open("../data/train_y", "rb"))
print "train done"
test_x = pickle.load(open("../data/test_x", "rb"))
test_y = pickle.load(open("../data/test_y", "rb"))
print "test done"

train_set = xgb.DMatrix(train_x, train_y)
validation_set = xgb.DMatrix(test_x, test_y)
watchlist = [(train_set, 'train'), (validation_set, 'eval')]
params = {"objective": 'multi:softmax',
           "booster" : "gbtree",
        'eval_metric':'merror',
                "eta": 0.06,
          "max_depth": 2,
          'num_class':2,
             'silent':1,
     'max_delta_step':5,
   'min_child_weight':3,
          'subsample':0.7,
              'gamma':1,
'early_stopping_rounds':10
}
rounds = 30
print "Training model..."
xgb_model = xgb.train(params, train_set, rounds, watchlist, verbose_eval=True)
train_pred = xgb_model.predict(xgb.DMatrix(test_x), ntree_limit=xgb_model.best_ntree_limit)
print train_pred

tp = 0
fn = 0
fp = 0
tn = 0
for i in range(len(test_y)):
    pred_y = int(train_pred[i])
    if test_y[i] == 1 and pred_y == 1:
        tp += 1
    elif test_y[i] == 1 and pred_y == 0:
        fn += 1
    elif test_y[i] == 0 and pred_y == 1:
        fp += 1
    elif test_y[i] == 0 and pred_y == 0:
        tn += 1

print tp, fn, fp, tn
test_ind = xgb_model.predict(xgb.DMatrix(test_x), ntree_limit=xgb_model.best_ntree_limit, pred_leaf=True)

train_ind = xgb_model.predict(xgb.DMatrix(train_x), ntree_limit=xgb_model.best_ntree_limit, pred_leaf=True)

print test_ind.shape, train_ind.shape
print test_ind
pickle.dump(test_ind, open("../data/gdbt_test_x_ind_1", "wb"))
pickle.dump(train_ind, open("../data/gdbt_train_x_ind_1", "wb"))
print "All done"
