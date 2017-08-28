#coding:utf-8
__author__ = 'liangnan03@meituan.com'

import pandas as pd
import numpy as np
from config import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import *

# 划分数据集
df = pd.read_csv(total_raw_features, sep='\t')
# df = pd.read_csv('./data/features/raw_features_20170728_ver1.csv', sep='\t')
feature = df.fillna(value=0).drop(['is_nobook', 'is_holiday_used', 'is_booth_used',
                                   'min_person_cnt', 'max_person_cnt', 'avg_person_cnt', 'interval_person_cnt',
                                   'item_index', 'page_city_id', 'geo_city_id', 'used_period',
                                   'mt_second_cate_id', 'mt_second_cate_name', 'mt_third_cate_id', 'mt_third_cate_name',
                                   'is_travel_master', 'cities_week_city', 'label'], axis=1)
feature = feature.ix[:, (feature != feature.ix[0]).any()]

X = feature.values
print len(feature.columns)
y = df['label'].values

#选择K个最好的特征，返回选择特征后的数据
# X = SelectKBest(chi2, k=30).fit_transform(X, y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# 5折交叉验证
AUC = []
Report = []
skf= StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, y):
    X_train,X_test = X[train_index], X[test_index]
    y_train,y_test = y[train_index], y[test_index]

    x_train= X_train
    x_test= X_test
    # 分类
    gbdt = GradientBoostingClassifier(learning_rate=0.1, n_estimators=500, max_depth=13, min_samples_leaf=17,
                                      min_samples_split=90, max_features=31, subsample=0.9, random_state=10)
    # gbdt = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7, min_samples_leaf =60,
    #            min_samples_split =1200, max_features='sqrt', subsample=0.8, random_state=10)
    gbdt.fit(x_train, y_train)
    y_pred = gbdt.predict(x_test)

    # 效果评估
    target_names = ['class 0', 'class 1']
    print classification_report(y_test, y_pred, target_names=target_names)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    auc_value = auc(fpr, tpr)
    print "\t\tauc\t\t",auc_value
    AUC.append(auc_value)

print "Average AUC : ", np.average(np.array(AUC))