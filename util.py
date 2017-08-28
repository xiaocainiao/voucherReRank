#coding:utf-8
__author__ = 'liangnan03@meituan.com'


import pandas as pd
from sklearn import metrics
import numpy as np
from config import *

def loadFeatureConfig():
    f = open(feature_config_path)
    features = {}
    for line in f.readlines():
        if not line:
            continue
        id_feature = line.split("\t")
        features[int(id_feature[0])] = id_feature[1].strip('\n')
    f.close()
    return features

def mapLRFeature(columns, weights, bias, min_max):
    registered_features = loadFeatureConfig()
    print registered_features
    print columns


    idlist = [k for column in columns for k, v in registered_features.iteritems() if v == column]
    min_max = {k:min_max[i] for i,k in enumerate(idlist)}

    print min_max

    output_weights = {}
    ids = ""
    for id,feature in registered_features.iteritems():
        if feature in columns:
            ids = ids + str(id) +","
            output_weights[int(id)] = weights[columns.index(feature)]

    print output_weights
    ids = ids[:-1]+"\n"

    if model_need_output:
        fout = open(model_config_file, 'a')
        fout.write("voucher_lr\n")
        fout.write(ids)
        fout.write("MIN_MAX\n")
        fout.write("b:"+str(bias[0])+'\n\n')

        for k,v in output_weights.iteritems():
            fout.write(str(k)+":"+str(v)+"\n")
        fout.write("\n")

        for k, v in min_max.iteritems():
            s = '%d:%s\t%s\n' % (k, v[0], v[1])
            fout.writelines(s)

        fout.close()


def checkMinMaxValue(columns):
    registered_features = loadFeatureConfig()
    print registered_features
    f = open(feature_min_max_file)
    min_max_values = {}
    for line in f.readlines():
        if line is None:
            continue
        id_values = line.split(":")
        min_max = id_values[1].strip('\n').split("\t")
        min_max[0] = float(min_max[0])
        min_max[1] = float(min_max[1])
        min_max_values[int(id_values[0])] = min_max
    f.close()

    ids = [k for column in columns for k,v in registered_features.iteritems() if v == column]
    min_max_values = [(k,min_max_values[k]) for k in sorted(min_max_values.keys(), key=ids.index)]

    return min_max_values


def min_max_scale(X, min_max):
    min = np.array([e[1][0] for e in min_max])
    max = np.array([e[1][1] for e in min_max])

    ret = []
    for x in X:
        x = (x - min)/(max - min)
        ret.append(x)

    return ret

def getTrainedWeights(columns):
    registered_features = loadFeatureConfig()

    weights = {}
    bias = []
    f = open(model_train_params)
    for line in f.readlines():
        if line is None:
            continue
        id_values = line.split(":")
        if id_values[0] == 'b':
            bias.append(float(id_values[1]))
        else:
            weights[int(id_values[0])] = float(id_values[1])

    ids = [k for column in columns for k, v in registered_features.iteritems() if v == column]
    weights = [(k, weights[k]) for k in sorted(weights.keys(), key=ids.index)]
    weights = np.array([e[1] for e in weights])
    bias = np.array(bias)

    return weights, bias

def sigmoid(X, W, b):
    return 1.0 / (1 + np.exp(-(np.matmul(X, W) + b)))


def split_map(x, a=0.3, b=0.7):
    return 0. if x < a else 1. if x > b else x


def extract_features(infile, degree=0.95):
    df = pd.read_csv(infile)
    obj = df.to_dict()
    fscores = obj['fscore']
    tot = sum(fscores.values())
    res = list()
    acc = 0
    for k in fscores.keys()[::-1]:
        acc += fscores[k]
        if float(acc) / float(tot) > degree:
            break
        res.append(obj['feature'][k])
    print len(res)
    print res


def calc_auc(df):
    y_true = df['label'].values
    y_pred = df['probability_label'].values
    auc = metrics.roc_auc_score(np.array(y_true), np.array(y_pred))
    return pd.DataFrame({'coupon_label': [df['coupon_label'][0]], 'auc': [auc]})


if __name__ == '__main__':
    registered_features = loadFeatureConfig()

