#coding:utf-8
__author__ = 'liangnan03@meituan.com'
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.ensemble import *
import pandas as pd
from config import *
from sklearn.preprocessing import *
from hyperopt import hp


def LRGridSearch(X_train, y_train):
    # 归一化
    if scale_method == 'standard_scale':
        sc = StandardScaler()
        sc.fit(X_train)
        x_train = sc.transform(X_train)
        # x_test = sc.transform(X_test)
    elif scale_method == 'normalize':
        nl = Normalizer()
        nl.fit(X_train)
        x_train = nl.transform(X_train)
        # x_test = nl.transform(X_test)
    else:
        mm = MinMaxScaler()
        mm.fit(X_train)
        x_train = mm.transform(X_train)
        # x_test = mm.transform(X_test)

    param_grid = {'C': range(1,2000,10)}
    clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
    clf = clf.fit(x_train, y_train)

    print clf.best_params_
    print clf.best_score_


def GBDTGridSearch(X_train, y_train):
    method
    # param_grid = {'n_estimators': range(30, 110, 10)}
    # param_grid ={'max_depth':range(5,14,2), 'min_samples_split':range(2,50,5)}
    # param_grid = {'min_samples_split': range(40, 100, 10), 'min_samples_leaf': range(1, 20, 2)}
    # param_grid = {'max_features': range(5, 39, 2)}
    param_grid = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
    clf = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=1,
                                                            n_estimators=100,
                                                            max_depth=13,
                                                            min_samples_split=90,
                                                            min_samples_leaf=17,
                                                            subsample=0.9,
                                                            max_features=31,
                                                            random_state=10),
                            param_grid=param_grid, scoring='roc_auc', iid=False, cv=5)
    clf = clf.fit(X_train, y_train)
    print clf.best_params_
    print clf.best_score_


def GBDTHyperopt(X_train, y_train):
    return X_train,y_train


if __name__ == '__main__':

    # 划分数据集
    df = pd.read_csv(total_raw_features, sep='\t')
    feature = df.fillna(value=0).drop(['is_nobook', 'is_holiday_used', 'is_booth_used',
                                       'min_person_cnt', 'max_person_cnt', 'avg_person_cnt', 'interval_person_cnt',
                                       'item_index', 'page_city_id', 'geo_city_id', 'used_period',
                                       'mt_second_cate_id', 'mt_second_cate_name', 'mt_third_cate_id',
                                       'mt_third_cate_name',
                                       'is_travel_master', 'cities_week_city', 'label'], axis=1)
    feature = feature.ix[:, (feature != feature.ix[0]).any()]

    X = feature.values
    print len(feature.columns)
    y = df['label'].values
    X_train = X
    y_train = y
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    if method == 'LR':
        LRGridSearch(X_train, y_train)
    else:
        GBDTGridSearch(X_train, y_train)
