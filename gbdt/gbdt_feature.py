#coding:utf-8

import xgboost as xgb
import cPickle as pickle
import Constants
import numpy as np


train_x = Constants.dir_path + "sample\\features\\training.gbdt.libfm"
train_y = np.loadtxt(Constants.dir_path + "sample\\training.Y", dtype=int)
print "train done"
test_x = Constants.dir_path + "sample\\features\\validation.gbdt.libfm"
test_y = np.loadtxt(Constants.dir_path + "sample\\validation.Y", dtype=int)
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
rounds = 20
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
pickle.dump(test_ind, open(Constants.dir_path + "sample\\features\\gbdt\\test.idx", "wb"))
pickle.dump(train_ind, open(Constants.dir_path + "sample\\features\\gbdt\\training.idx", "wb"))
print "All done"
