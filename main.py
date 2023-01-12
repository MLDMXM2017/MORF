from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import numpy as np
import pandas as pd
import random
from logger import get_logger
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier

from segmentor import *
from mvhh_n import *
from mvhh_r import *


np.seterr(divide='ignore',invalid='ignore')

#*********选择算法*********
def make_estimator(method='random_forest', view=[[0,199]], max_depth=50, n_estimators=100):
    if  method == 'mvhh(n)':
        return MVHH_N_Classifier(Entropy(),MeanSegmentor(),view=view, max_depth = max_depth)

    elif method == 'mvhh(n)_bag':
        return BaggingClassifier(base_estimator=MVHH_N_Classifier(Entropy(),MeanSegmentor(),view=view, max_depth = max_depth),
                                 n_estimators=n_estimators)
    elif method == 'mvhh(r)':
        return MVHH_R_Classifier(Entropy(),MeanSegmentor(), view=view,max_depth = max_depth)

    elif method == 'mvhh(r)_bag':
        return BaggingClassifier(base_estimator=MVHH_R_Classifier(Entropy(),MeanSegmentor(),view=view, max_depth = max_depth),
                                 n_estimators=n_estimators)
    else:
        ValueError('Unknown model!')

#*********调整*********
method_type='mvhh(r)'
f_type = "ECFP2"

LOGGER = get_logger("-", 'May-10')
LOGGER.info("====================Version6====================")

#*********选择数据*********
def load_data(feature_type="MACCS",if_pd=False):
    labels = pd.read_csv(label_file, index_col=0)
    finger_file = None
    if feature_type == "ECFP2":
        finger_file = finger1_file

    target = target_file
    phychem = phychem_file
    exp = exp_file

    target_pd = pd.read_csv(target, index_col=0)
    phychem_pd = pd.read_csv(phychem, index_col=0)
    finger_pd = pd.read_csv(finger_file, index_col=0)
    exp_pd = pd.read_csv(exp, index_col=0)

    # 4个的
    # view = [[0] * 2 for i in range(4)]
    # view[0][1] = finger_pd.shape[1] - 1
    # view[1][0] = view[0][1] + 1
    # view[1][1] = view[1][0] + phychem_pd.shape[1] - 1
    # view[2][0] = view[1][1] + 1
    # view[2][1] = view[2][0] + target_pd.shape[1] - 1
    # view[3][0] = view[2][1] + 1
    # view[3][1] = view[3][0] + exp_pd.shape[1] - 1
    # print('VIEW:',view)
    # name = []
    # for i in range(0, view[3][1] + 1):
    #     name.append(str(i))
    # x = pd.concat([finger_pd, phychem_pd], axis=1)
    # x = pd.concat([x, target_pd], axis=1)
    # x = pd.concat([x, exp_pd], axis=1)
    # x.columns = name

    # 3个的
    # flag1 = phychem_pd
    # flag2 = target_pd
    # flag3 = exp_pd
    # view = [[0] * 2 for i in range(3)]
    # view[0][1] = flag1.shape[1] - 1
    # view[1][0] = view[0][1] + 1
    # view[1][1] = view[1][0] + flag2.shape[1] - 1
    # view[2][0] = view[1][1] + 1
    # view[2][1] = view[2][0] + flag3.shape[1] - 1
    # print('VIEW:',view)
    # name = []
    # for i in range(0, view[2][1] + 1):
    #     name.append(str(i))
    # x = pd.concat([flag1, flag2], axis=1)
    # x = pd.concat([x, flag3], axis=1)
    # x.columns = name

    # 2个的
    flag1 = finger_pd
    flag2 = phychem_pd
    view = [[0] * 2 for i in range(2)]
    view[0][1] = flag1.shape[1] - 1
    view[1][0] = view[0][1] + 1
    view[1][1] = view[1][0] + flag2.shape[1] - 1
    print('VIEW:',view)
    name = []
    for i in range(0, view[1][1] + 1):
        name.append(str(i))
    x = pd.concat([flag1, flag2], axis=1)
    x.columns = name

    #1个的：
    # x = finger_pd
    # name = []
    # view = [[0,x.shape[1] - 1]]
    # for i in range(0, x.shape[1]):
    #     name.append(str(i))
    # x.columns = name

    # *********去除数据同标签不同的异常*********
    #找出异常
    name2=[]
    xs =finger_pd
    for i in range(0,xs.shape[1]):
        name2.append((str(i)))
    xs.columns=name2
    xs = pd.concat([xs, labels], axis=1)
    df_cat = xs
    df_nodup = df_cat.drop_duplicates()#原始去重
    df_nodup4 = df_nodup.drop_duplicates(subset=name2, keep=False)#去重之后再去X相同的重，也就是有异常的了
    data = df_nodup4.append(df_nodup).drop_duplicates(keep=False)#找出这些异常

    #去除异常
    xy = pd.concat([x, labels], axis=1)#原始数据
    df_over = xy.drop(index=data.index)#在原始数据中去除这些异常

    x = df_over[name]
    print(x.shape)
    y=df_over['Y']
    x, y = np.array(x), np.array(y)

    return x, y, view

# 分子指纹
finger1_file = "data/V6/fp.csv"
# 物化性质
phychem_file = "data/V6/phychem.csv"
# 靶标蛋白
target_file = "data/V6/target.csv"
# 表达谱
exp_file = "data/V6/expression.csv"
# label
label_file = "data/V6/label_CID.csv"

x, y ,view= load_data(feature_type=f_type)
x, y = np.array(x), np.array(y)
for j in range(x.shape[1]):
    min_value = abs(min(x[:, j]))
    x[:, j] += min_value

y = pd.DataFrame(y).values.ravel()
LOGGER.info("data shape: {}".format(x.shape))
accs, f1s, aucs, recall_0s, recall_1s, precision_0s, precision_1s, issame = [], [], [], [], [], [], [], []

#*********执行算法*********
def changeData(x_train,x_test,y_train,y_test,view):
    X_real = None
    X_real_test = None
    viewlen = []
    # 分不同的view来选择一部分特征
    # 如果特征数大于300，且满足卡方检验操作的条件，选择50%
    # 在V6中expression不能做卡方检验，设置i，跳过
    for i in range(0, len(view)):
        start = view[i][0]
        end = view[i][1]
        feature_ids = range(start, end + 1)
        sub = x_train[:, feature_ids]

        if ((end + 1 - start) > 300) and i!=8:
            change = SelectPercentile(chi2, percentile=50).fit(sub, y_train)
            X_new = change.transform(sub)
        else:
            X_new = sub
        X_new = pd.DataFrame(X_new)
        viewlen.append(X_new.shape[1])
        if i==0 :
            X_real = X_new
        else:
            X_real = pd.concat([X_real, X_new],axis=1)

        sub_test = x_test[:, feature_ids]
        if ((end + 1 - start) > 300) and i!=8:
            X_test_new = change.transform(sub_test)
        else:
            X_test_new = sub_test
        X_test_new = pd.DataFrame(X_test_new)
        if i == 0:
            X_real_test=X_test_new
        else:
            X_real_test = pd.concat([X_real_test, X_test_new],axis=1)

    #4个的：
    # view = [[0] * 2 for j in range(4)]
    # view[0][1] = viewlen[0] - 1
    # view[1][0] = view[0][1] + 1
    # view[1][1] = view[1][0] + viewlen[1] - 1
    # view[2][0] = view[1][1] + 1
    # view[2][1] = view[2][0] + viewlen[2] - 1
    # view[3][0] = view[2][1] + 1
    # view[3][1] = view[3][0] + viewlen[3] - 1
    # name = []
    # for z in range(0, view[3][1] + 1):
    #     name.append(str(z))

    #3个的：
    # view = [[0] * 2 for j in range(3)]
    # view[0][1] = viewlen[0] - 1
    # view[1][0] = view[0][1] + 1
    # view[1][1] = view[1][0] + viewlen[1] - 1
    # view[2][0] = view[1][1] + 1
    # view[2][1] = view[2][0] + viewlen[2] - 1
    # name = []
    # for z in range(0, view[2][1] + 1):
    #     name.append(str(z))

    # 2个的
    view = [[0] * 2 for j in range(2)]
    view[0][1] = viewlen[0] - 1
    view[1][0] = view[0][1] + 1
    view[1][1] = view[1][0] + viewlen[1] - 1
    name = []
    for z in range(0, view[1][1] + 1):
        name.append(str(z))

    # 1个的：
    # view = [[0,viewlen[0] - 1]]
    # name = []
    # for z in range(0, view[0][1] + 1):
    #     name.append(str(z))
    #
    X_real.columns = name
    X_real_test.columns = name

    return np.array(X_real),np.array(X_real_test),view


LOGGER.info("====================Method_{}====================".format(method_type))

k=1
for h in range(0,1):
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0)

    for train_id, test_id in skf.split(x, y):
        LOGGER.info("----------Stratified Kfold-{}----------".format(k))
        x_train, x_test = x[train_id], x[test_id]
        y_train, y_test = y[train_id], y[test_id]

        view_real = view
        # 选择特征（可选步骤）
        x_train, x_test, view_real = changeData(x_train, x_test, y_train, y_test, view)
        print('VIEW_real:', view_real)
        # 分类器
        clf = make_estimator(method=method_type, view=view_real)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        # 结果统计
        report = classification_report(y_test, y_pred, output_dict=True)
        acc = report['accuracy']
        f1 = report['macro avg']['f1-score']
        recall_0 = report["0"]['recall']
        recall_1 = report["1"]['recall']
        precision_0 = report["0"]['precision']
        precision_1 = report["1"]['precision']

        accs.append(acc)
        f1s.append(f1)
        recall_0s.append(recall_0)
        recall_1s.append(recall_1)
        precision_0s.append(precision_0)
        precision_1s.append(precision_1)
        k += 1

LOGGER.info("accuracy mean={:.6f}, std={:.6f}".format(np.mean(accs), np.std(accs)))
LOGGER.info("f1_score mean={:.6f}, std={:.6f}".format(np.mean(f1s), np.std(f1s)))
LOGGER.info("recall_0 mean={:.6f}, std={:.6f}".format(np.mean(recall_0s), np.std(recall_0s)))
LOGGER.info("recall_1 mean={:.6f}, std={:.6f}".format(np.mean(recall_1s), np.std(recall_1s)))
LOGGER.info("precision_0 mean={:.6f}, std={:.6f}".format(np.mean(precision_0s), np.std(precision_0s)))
LOGGER.info("precision_1 mean={:.6f}, std={:.6f}".format(np.mean(precision_1s), np.std(precision_1s)))

print(round(np.mean(accs),6),' ',
      round(np.mean(f1s),6),' ',round(np.mean(recall_0s),6),' ',round(np.mean(recall_1s),6),' ',
      round(np.mean(precision_0s),6),' ',round(np.mean(precision_1s),6),' ',
      round(np.std(accs),6),' ',
      round(np.std(f1s),6),' ',round(np.std(recall_0s),6),' ',round(np.std(recall_1s),6),' ',
      round(np.std(precision_0s),6),' ',round(np.std(precision_1s),6))