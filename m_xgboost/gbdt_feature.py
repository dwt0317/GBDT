#coding:utf-8

import numpy as np
from sklearn import metrics   #Additional scklearn functions
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from xgboost.sklearn import XGBClassifier

import Constants
import xgboost as xgb


# tp, fn, fp, tn
def get_metric(test_y, train_pred):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(len(test_y)):
        pred_y = 0
        if train_pred[i] > 0.5:
            pred_y = 1
        if test_y[i] == 1 and pred_y == 1:
            tp += 1
        elif test_y[i] == 1 and pred_y == 0:
            fn += 1
        elif test_y[i] == 0 and pred_y == 1:
            fp += 1
        elif test_y[i] == 0 and pred_y == 0:
            tn += 1
    print tp, fn, fp, tn

def train_model():
    train_x = Constants.dir_path + "sample\\features\\training.gbdt.libfm"
    train_y = np.loadtxt(Constants.dir_path + "sample\\training.Y", dtype=int)
    test_x = Constants.dir_path + "sample\\features\\test.gbdt.libfm"
    test_y = np.loadtxt(Constants.dir_path + "sample\\test.Y", dtype=int)


    train_data = load_svmlight_file(train_x)

    rounds = 20
    classifier = XGBClassifier(learning_rate=0.1, n_estimators=rounds, max_depth=3,
                               min_child_weight=1, gamma=0, subsample=0.8,
                               objective='binary:logistic', nthread=2)

    grid = False
    if grid:
        param_test1 = {
            'max_depth': range(3, 5, 2),
            'min_child_weight': range(1, 6, 3)
        }
        gsearch = GridSearchCV(estimator=classifier, param_grid=param_test1, scoring='roc_auc', n_jobs=2)
        gsearch.fit(train_data[0].toarray(), train_data[1])
        print gsearch.best_params_, gsearch.best_score_

    if not grid:
        train_set = xgb.DMatrix(train_x)
        print "train done"
        validation_set = xgb.DMatrix(test_x)
        print "test done"
        watchlist = [(train_set, 'train'), (validation_set, 'eval')]
        params = {"objective": 'binary:logistic',
                  "booster": "gbtree",
                  'eval_metric': 'auc',
                  "eta": 0.1,
                  "max_depth": 3,
                  'silent': 0,
                  'min_child_weight': 1,
                  'subsample': 0.8,
                  'gamma': 0,
                  'early_stopping_rounds': 10,
                  'nthread': 2
                  }
        print "Training model..."
        xgb_model = xgb.train(params, train_set, rounds, watchlist, verbose_eval=True)
        train_pred = xgb_model.predict(xgb.DMatrix(test_x))
        print train_pred
        auc_test = metrics.roc_auc_score(test_y, train_pred)
        print auc_test

    # test_ind = xgb_model.predict(xgb.DMatrix(test_x), ntree_limit=xgb_model.best_ntree_limit, pred_leaf=True)
    # train_ind = xgb_model.predict(xgb.DMatrix(train_x), ntree_limit=xgb_model.best_ntree_limit, pred_leaf=True)
    #
    # print test_ind.shape, train_ind.shape
    # for i in range(30):
    #     print test_ind[i]
    # # pickle.dump(test_ind, open(Constants.dir_path + "sample\\features\\m_gbdt\\test.idx", "wb"))
    # # pickle.dump(train_ind, open(Constants.dir_path + "sample\\features\\m_gbdt\\training.idx", "wb"))
    # print "All done"


if __name__ == '__main__':
    train_model()

