#coding:utf-8
__author__ = 'liangnan03@meituan.com'

import pandas as pd
import numpy as np
import os
import config
from dataView import *
from util import *
from sklearn.model_selection import StratifiedShuffleSplit

def getTotalSamples(data):
    # 去重
    print "before drop duplicates...",len(data)
    data = data.drop_duplicates()
    print "after drop duplicates...", len(data)

    # 获取正样本
    pos_data = data[data['label'] == 1]
    total_pos_samples_num = len(pos_data)

    # 采样负样本 负样本数目:正样本数目 = config.neg_pos_samples_ratio
    total_neg_samples_num = int(total_pos_samples_num*config.neg_pos_samples_ratio)

    # 清除NaN值过多的样本
    neg_data = data[data['label'] == 0]
    neg_data = neg_data[neg_data.isnull().sum(axis=1) <= 20]
    print "negative data after cleaning NaN...", len(neg_data)

    # 随机采样
    neg_data = neg_data.sample(total_neg_samples_num)

    # 合并正负样本
    print "positive samples...", len(pos_data)
    print "negative samples...", len(neg_data)
    data = pd.concat([pos_data, neg_data], ignore_index=True)
    data = data.drop(['preference_workday','preference_weekend'], axis=1)
    return data

def missingValueProcessor(data):
    #TODO
    print data.columns

    for column in data.columns:
        # print column,type(data[column][0])
        if type(data[column][0]) == np.str:
            data[column].fillna(value="PAD")
        elif type(data[column][0]) == np.float64:
            data[column].fillna(value=0.0)

    data = data.drop(['user_id', 'deal_id', 'partition_date'], axis=1)
    data = data.dropna(how='all', axis=1)
    data = data.ix[:, (data != data.ix[0]).any()]

    return data

if __name__ == '__main__':
    files = ['vrr_samples_20170721.xlsx', 'vrr_samples_20170722.xlsx', 'vrr_samples_20170723.xlsx', 'vrr_samples_20170724.xlsx', 'vrr_samples_20170725.xlsx',
             'vrr_samples_20170726.xlsx', 'vrr_samples_20170727.xlsx', 'vrr_samples_20170728.xlsx', 'vrr_samples_20170729.xlsx', 'vrr_samples_20170730.xlsx',
             'vrr_samples_20170731.xlsx', 'vrr_samples_20170801.xlsx', 'vrr_samples_20170802.xlsx', 'vrr_samples_20170803.xlsx', 'vrr_samples_20170804.xlsx',
             'vrr_samples_20170805.xlsx', 'vrr_samples_20170806.xlsx', 'vrr_samples_20170807.xlsx', 'vrr_samples_20170808.xlsx', 'vrr_samples_20170809.xlsx',
             'vrr_samples_20170810.xlsx', 'vrr_samples_20170811.xlsx', 'vrr_samples_20170812.xlsx', 'vrr_samples_20170813.xlsx', 'vrr_samples_20170814.xlsx',
             'vrr_samples_20170815.xlsx', 'vrr_samples_20170816.xlsx', 'vrr_samples_20170817.xlsx', 'vrr_samples_20170818.xlsx', 'vrr_samples_20170819.xlsx',
             'vrr_samples_20170820.xlsx', 'vrr_samples_20170821.xlsx', 'vrr_samples_20170822.xlsx', 'vrr_samples_20170823.xlsx', 'vrr_samples_20170824.xlsx',
             'vrr_samples_20170825.xlsx', 'vrr_samples_20170826.xlsx', 'vrr_samples_20170827.xlsx', 'vrr_samples_20170828.xlsx', 'vrr_samples_20170829.xlsx',
             'vrr_samples_20170830.xlsx', 'vrr_samples_20170831.xlsx', 'vrr_samples_20170901.xlsx', 'vrr_samples_20170902.xlsx', 'vrr_samples_20170903.xlsx']

    Data = pd.DataFrame()

    if not os.path.exists(config.total_samples):
        for file_name in files:
            path = './data/timeShiftSamples/'+file_name
            dv = DataView(path)
            data = getTotalSamples(dv.data)
            print "samples size of "+file_name, len(data)
            Data = pd.concat([Data, data], ignore_index=True)

        Data.to_csv(config.total_samples, sep='\t', encoding='utf-8', index=False)

    df = pd.read_csv(config.total_samples, sep='\t')


    # 缺失值处理
    df = missingValueProcessor(df)

    # 特征选择
    # features= loadFeatureConfig()
    # involved_features = list(features.values())
    # print df.columns()
    # df = df[involved_features]

    df.to_csv(config.total_raw_features, sep='\t', encoding='utf-8', index=False)















