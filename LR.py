#coding:utf-8
__author__ = 'liangnan03@meituan.com'

from util import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold

def getDataSet():
    # 划分数据集
    df = pd.read_csv(total_raw_features, sep='\t')

    # 丢弃目前不使用的特征
    feature = df.fillna(value=0).drop(['is_nobook','is_holiday_used','is_booth_used',
                                       'min_person_cnt','max_person_cnt','avg_person_cnt','interval_person_cnt',
                                       'item_index','page_city_id','geo_city_id','used_period',
                                       'mt_second_cate_id','mt_second_cate_name','mt_third_cate_id','mt_third_cate_name',
                                       'is_travel_master','cities_week_city','label'], axis=1)
    feature = feature.ix[:, (feature != feature.ix[0]).any()]

    X = feature.values
    print sorted(X[:,0])[-400]
    print len(feature.columns)
    y = df['label'].values

    #选择K个最好的特征，返回选择特征后的数据
    # X = SelectKBest(chi2, k=30).fit_transform(X, y)
    return X, y, feature

# 10折交叉验证
def evaluation():
    AUC = []
    skf= StratifiedKFold(n_splits=10, shuffle=True)

    for train_index, test_index in skf.split(X, y):
        X_train,X_test = X[train_index], X[test_index]
        y_train,y_test = y[train_index], y[test_index]

        # 归一化
        if scale_method == 'standard_scale':
            sc = StandardScaler()
            sc.fit(X_train)
            x_train = sc.transform(X_train)
            x_test = sc.transform(X_test)
        elif scale_method == 'normalize':
            # p-范数的计算公式：||X||p=(|x1|^p+|x2|^p+...+|xn|^p)^1/p
            # 默认l2，求每个样本的二范数，每个特征值/二范数
            nl = Normalizer()
            nl.fit(X_train)
            x_train = nl.transform(X_train)
            x_test = nl.transform(X_test)
        else:
            mm = MinMaxScaler()
            mm.fit(X_train)
            x_train = mm.transform(X_train)
            x_test = mm.transform(X_test)

        # 分类
        lr = LogisticRegression(C=51)
        lr.fit(x_train, y_train)
        y_pred = lr.predict_proba(x_test)[:,1]
        # y_pred = lr.predict_proba(x_test)


        # 效果评估
        # target_names = ['class 0', 'class 1']
        # print classification_report(y_test, y_pred, target_names=target_names)

        fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        auc_value = auc(fpr, tpr)
        print "\t\tauc\t\t",auc_value
        AUC.append(auc_value)

    print "Average AUC : ", np.average(np.array(AUC))

def training(X, y):
    X_train = X; y_train = y

    # 归一化
    if scale_method == 'standard_scale':
        sc = StandardScaler()
        sc.fit(X_train)
        x_train = sc.transform(X_train)
    elif scale_method == 'normalize':
        # p-范数的计算公式：||X||p=(|x1|^p+|x2|^p+...+|xn|^p)^1/p
        # 默认l2，求每个样本的二范数，每个特征值/二范数
        nl = Normalizer()
        nl.fit(X_train)
        x_train = nl.transform(X_train)
    else:
        mm = MinMaxScaler()
        mm.fit(X_train)
        x_train = mm.transform(X_train)

    lr = LogisticRegression(C=51)
    lr.fit(x_train, y_train)
    from sklearn.externals import joblib
    joblib.dump(lr, 'lr.model')

    return lr.coef_, lr.intercept_


def predictWithGlobalMinMax(X, y):
    MIN_MAX = checkMinMaxValue(feature.columns)

    X = min_max_scale(X, MIN_MAX)
    W, b = getTrainedWeights(feature.columns)
    prob = sigmoid(X, W, b)
    y_pred = [1 if p >= 0.5 else 0 for p in prob]

    target_names = ['class 0', 'class 1']
    print classification_report(y, y_pred, target_names=target_names)

    fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=1)
    auc_value = auc(fpr, tpr)
    print "\t\tauc\t\t",auc_value



if __name__ == '__main__':
    X, y, feature = getDataSet()
    fs = list(feature.columns)

    if procedure == 'evaluation':
        evaluation()
    elif procedure == 'train':
        w, b = training(X, y)
        min_max = []
        for f in np.array(X).transpose():
            min_max.append([min(f), max(f)])
        mapLRFeature(columns=fs, weights=list(w[0]), bias=list(b), min_max=min_max)
    elif procedure == 'predict':
        predictWithGlobalMinMax(X, y)





