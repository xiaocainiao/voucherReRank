#coding:utf-8
__author__ = 'liangnan03@meituan.com'

# data file path
file_path = './data/timeShiftSamples/vrr_samples_20170810.xlsx'

total_samples = './data/dataset/samples.csv'

total_raw_features = './data/features/raw_features.csv'


# model path
model_need_output = False
model_path = './model'
model_config_file = './model/output_config_file.txt'
model_train_params = 'weights.txt'


# dataset split params
neg_pos_samples_ratio = 1.5


# model
method = 'LR'
# method = 'GBDT'
# feature
feature_config_path = 'featureConfig.txt'
feature_min_max_file = 'MIN_MAX.txt'


# scaling
scale_method = 'max_min_scale'
# scale_method = 'normalize'
# scale_method = 'standard_scale'


# training
# procedure = 'train'
procedure = 'evaluation'
# procedure = 'predict'


