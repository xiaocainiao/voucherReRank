# voucherReRank

# 数据
data/timeShiftSamples/* 中存放的是raw samples
data/dataset/samples.csv 中存放合并采样后的样本数据集
data/features/raw_features 中存放过滤了部分无用字段后的样本特征数据

# 模型
model/output_config_file.txt 是我们定义好格式的模型输出文件
lr.model为sci-kit中LR模型的输出文件

# 代码
配置config.py中neg_pos_samples_ratio，作为负样本/正样本的比值，对负样本进行下采样
运行dataSplit.py对每日样本进行合并、采样处理
运行GridSearch.py可进行网格搜索，调整参数的取值，包括对LR和GBDT的调参，其中GBDT的五行调参是逐行分步进行的
配置config.py中method，选择一个模型进行调参
    method='LR'时，调用LR的调参函数
    method='GBDT'时，调用GBDT的调参函数

运行LR.py读取数据集可进行交叉验证、训练、预测等，不同任务的选择要在config.py中配置procedure的取值
    procedure='evaluation'时，输出离线训练模型的10（5）倍交叉验证AUC
    procedure='train'时，使用所有数据构造输出文件，在config.py中配置model_need_output=True时，模型文件会输出，否则只打印
运行GBDT.py可进行树模型的离线评估

运行onlineRankResult.py可调用线上接口，查看指定用户的排序列表
运行casePredictAnalysis.py可验证当前线上使用模型的参数是否正确
    分别输出通过sigmoid函数预测的回归值和通过load sci-kit输出文件lr.model预测的回归值

NOTE:模型输出文件output_config_file.txt中，49号特征为距离，单位为千米，工程中单位为米，因此需要对输出文件中49号特征的最大最小值扩大1000倍
