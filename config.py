#coding:utf-8
__author__ = 'liangnan03@meituan.com'

# data file path
# file_path = './data/vrr_samples_v1.txt'
file_path = './data/vrr_distinct_samples.xlsx'

total_samples = './data/dataset/samples.csv'
total_raw_features = './data/features/raw_features.csv'


train_path = './data/data_split/train_data/'
train_feature_data_path = train_path + 'features/'
train_raw_data_path = train_path + 'raw_data.csv'


validate_path = './data/data_split/validate_data/'
validate_feature_data_path = validate_path + 'features/'
validate_raw_data_path = validate_path + 'raw_data.csv'

predict_path = './data/data_split/predict_data/'
predict_feature_data_path = predict_path + 'features/'
predict_raw_data_path = predict_path + 'raw_data.csv'


# model path
model_need_output = False
model_path = './model'
model_config_file = './model/output_config_file.txt'
model_train_params = 'weights.txt'
model_feature_importance_file = '/feature_importance.png'
model_feature_importance_csv = '/feature_importance.csv'
model_train_log = '/train.log'
model_params = '/param.json'


# dataset split params
neg_pos_samples_ratio = 1.5





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


