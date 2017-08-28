#coding:utf-8
__author__ = 'liangnan03@meituan.com'

import json
import pandas as pd
import numpy as np
import os
from util import *
from sklearn.model_selection import StratifiedShuffleSplit



if __name__ == '__main__':
    file = open("caseFeature.txt",'r')

    user_and_user_deal = file.readline()

    # get user feature
    user = user_and_user_deal[15:71]
    user_feature = {}
    for p in user.split(','):
        kv = p.split(':')
        user_feature[int(kv[0])] = float(kv[1])

    # get user-deal feature
    user_deal_id = [21, 22, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36]
    user_deal_feature = {}
    for e in user_deal_id:
        user_deal_feature[e] = 0.0

    # get deal feature and construct all features
    Features = []
    for line in file.readlines():
        if line != '\n':
            features = {}
            deal_feature = {}
            line = line[1:-2].split(',')
            for e in line:
                kv = e.split(':')
                deal_feature[int(kv[0])] = float(kv[1])
            if(len(deal_feature) == 39 or len(deal_feature) == 1):
                features = deal_feature
            else:
                features = dict(user_feature.items()+user_deal_feature.items()+deal_feature.items())
            Features.append(features)

    print Features

    # used features' id
    used_feature_id = [1,2,3,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,49]

    # get model weights
    w = open("weights.txt", 'r')
    b = float(w.readline()[:-1].split(':')[1])
    print "b..."
    print b
    weights = [0.0] * 39
    for line in w.readlines():
        line = line[:-1]
        kv = line.split(':')
        if int(kv[0]) == 49:
            weights[0] = float(kv[1])
        else:
            weights[used_feature_id.index(int(kv[0]))+1] = float(kv[1])
    weight = np.array(weights)

    # get model min_max value
    m = open("MIN_MAX.txt", 'r')
    min_max_values = {}
    for line in m.readlines():
        if line is None:
            continue
        id_values = line.split(":")
        min_max = id_values[1].strip('\n').split("\t")
        min_max[0] = float(min_max[0])
        min_max[1] = float(min_max[1])
        min_max_values[int(id_values[0])] = min_max
    MM = [[0.0,1.0]]*39
    for k, v in min_max_values.iteritems():
        if k == 49:
            MM[0] = v
        else:
            MM[used_feature_id.index(int(k))+1] = v

    # map to feature matrix
    F = []
    for features in Features:
        f = [0.0] * 39
        for k in features.keys():
            if k in used_feature_id:
                value = features[k]
                if k == 49:
                    f[0] = value
                else:
                    # value = (features[k] - MM[used_feature_id.index(k)][0])/(MM[used_feature_id.index(k)][1]-MM[used_feature_id.index(k)][0])
                    f[used_feature_id.index(k) + 1] = value
        F.append(f)
    F = np.array(F)
    print F
    XX = F/(F[-1,:])
    print XX

    p = sigmoid(XX, weight, b)
    print list(p)

    from sklearn.externals import joblib
    from sklearn.linear_model import *
    from sklearn.preprocessing import *

    lr = joblib.load('lr.model')
    sc = MinMaxScaler()
    sc.fit(X=F)
    F = sc.transform(F)

    pp = LogisticRegression.predict_proba(lr, F)
    print [e[1] for e in pp]




















