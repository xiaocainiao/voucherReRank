#coding:utf-8
__author__ = 'liangnan03@meituan.com'

import pandas as pd
import numpy as np
from config import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
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

    # 对训练集再进行划分
    x_train, x_train_lr, y_train, y_train_lr = train_test_split(x_train, y_train, test_size=0.5)

    # 调用one-hot编码。
    grd_enc = OneHotEncoder()

    # 调用LR分类模型。
    grd_lr = LogisticRegression()

    gbdt.fit(x_train, y_train)

    # fit one-hot编码器 apply()返回的多维数组 shape = [n_samples, n_estimators, n_classes]，二分类中n_classes=1
    grd_enc.fit(gbdt.apply(x_train)[:, :, 0])

    # 使用训练好的GBDT模型构建特征，然后将特征经过one-hot编码作为新的特征输入到LR模型训练。
    grd_lr.fit(grd_enc.transform(gbdt.apply(x_train_lr)[:, :, 0]), y_train_lr)

    # 用训练好的LR模型多X_test做预测
    y_pred = grd_lr.predict_proba(grd_enc.transform(gbdt.apply(x_test)[:, :, 0]))[:, 1]
    # y_pred = grd_lr.predict(grd_enc.transform(gbdt.apply(x_test)[:, :, 0]))

    # 效果评估
    # target_names = ['class 0', 'class 1']
    # print classification_report(y_test, y_pred, target_names=target_names)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    auc_value = auc(fpr, tpr)
    print "\t\tauc\t\t",auc_value
    AUC.append(auc_value)

print "Average AUC : ", np.average(np.array(AUC))