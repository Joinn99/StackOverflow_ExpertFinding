# analysis.py
# 模型性能测试与分析
# python3.x

import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
from model.utils import load_data, get_tags

def best_solution(tag):
    """
    在数据库中寻找话题标签[TAG]下最佳实验结果及对应的参数设置
    """
    info = sqlite3.connect('Data/info.sqlite')
    par = pd.read_sql_query("""
        SELECT * FROM XGBClassifier JOIN(
        SELECT MAX(f1) mf1 FROM XGBClassifier WHERE tag='{tag}')
        ON f1=mf1 AND tag='{tag}'
        """.format(**{'tag': tag}), info)
    info.close()
    return par.drop(columns=['id', 'precision', 'recall', 'f1'])


def find_expert(tag):
    """
    输出话题标签[TAG]下模型预测的最有可能是潜在专家的20名用户
    """
    fold = StratifiedKFold(n_splits=4)
    params = best_solution(tag)
    data, target, ratio = load_data(tag)
    fold.random_state = int(params['seed'])
    samp = ADASYN(n_neighbors=2,
                  sampling_strategy=float(params['sampling_strategy']) * ratio,
                  random_state=int(params['seed']))
    clf = XGBClassifier(
        n_estimators=int(params['n_estimators']),
        gamma=float(params['gamma']),
        eta=float(params['eta']),
        reg_lambda=int(params['reg_lambda']),
        verbosity=0,
        n_jobs=-1,
        random_state=int(params['seed']))
    pipeline = Pipeline(
        [(type(samp).__name__, samp), (type(clf).__name__, clf)])
    experts = pd.DataFrame(columns=['id', 'probability'])
    for _, (train, test) in tqdm(enumerate(fold.split(data, target)), total=4):
        pipeline.fit(data.iloc[train], target.iloc[train])
        pred_proba = pd.Series(pipeline.predict_proba(data.iloc[test])[:, 1],
                               index=target.iloc[test].index, name='probability')
        experts = experts.append(pred_proba.to_frame().reset_index())
    experts = experts.sort_values(by=['probability'], ascending=False).iloc[:20]
    experts['probability'] = experts['probability'].astype(float).map("{:.1%}".format)
    print(experts.to_string(index=False))


def display(y_real, y_proba, f_importance, col):
    """
    绘制准确率召回率曲线，输出特征重要性
    """
    plt.style.use('ggplot')
    plt.rc('font', family='Times New Roman', weight='bold', size=12)
    _, axes = plt.subplots(1, 1, figsize=(10, 5))
    for f_score in np.linspace(0.55, 0.75, num=5):
        var_x = np.linspace(0.01, 1)
        var_y = f_score * var_x / (2 * var_x - f_score)
        _, = plt.plot(var_x[var_y >= 0], var_y[var_y >= 0], '--', color='red',
                      alpha=0.5, lw=1, label='ISO F1 Curves')
        plt.annotate('f1={0:0.2f}'.format(f_score), xy=(0.9, var_y[45] + 0.02))

    for i in range(6):
        y_real[i] = np.concatenate(y_real[i])
        y_proba[i] = np.concatenate(y_proba[i])
        precision, recall, _ = precision_recall_curve(y_real[i], y_proba[i])
        axes.step(recall, precision, lw=2, alpha=0.3,
                  label='Fold %d AUC=%.4f' % (i+1, auc(recall, precision)))
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    f_importance = np.mean(np.array(f_importance), axis=0)
    tqdm.write("{st}Feature Importance{st}".format(**{"st": "*"*15}))
    tqdm.write("\n".join(["{}: {}".format(k, v) for (k, v) in zip(
        col, list(f_importance))]))
    tqdm.write("*"*50)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    axes.step(recall, precision, lw=2, color='black', alpha=0.9,
              label='Overall AUC=%.4f' % (auc(recall, precision)))
    axes.set_title('Precision Recall Curve')
    axes.set_xlabel('Recall')
    axes.set_ylabel('Precision')
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    axes.legend(handles, labels, loc='lower left', fontsize='small')
    plt.show()


def performance():
    """
    分析模型性能
    """
    tqdm.write("*"*50 + "\n\tStackOverlflow Expert Prediction\n"+ "*"*50)
    val = RepeatedStratifiedKFold(n_splits=3, n_repeats=2)
    y_real = [[], [], [], [], [], []]
    y_proba = [[], [], [], [], [], []]
    f_importance = []
    # 对每个话题标签分别进行测试
    for tag in tqdm(get_tags()):
        params = best_solution(tag)             # 获取最优参数
        data, target, ratio = load_data(tag)    # 加载数据
        val.random_state = int(params['seed'])  # 设置随机数种子
        # 建立过采样和分类器的流水线模型
        samp = ADASYN(n_neighbors=2,
                      sampling_strategy=float(
                          params['sampling_strategy']) * ratio,
                      random_state=int(params['seed']))
        clf = XGBClassifier(
            n_estimators=int(params['n_estimators']),
            gamma=float(params['gamma']),
            eta=float(params['eta']),
            reg_lambda=int(params['reg_lambda']),
            verbosity=0,
            n_jobs=-1,
            random_state=int(params['seed'])
        )
        pipeline = Pipeline(
            [(type(samp).__name__, samp), (type(clf).__name__, clf)])
        # 对交叉验证的子集分别进行测试
        for ind, (train, test) in tqdm(enumerate(val.split(data, target)), leave=False, total=6):
            pipeline.fit(data.iloc[train], target.iloc[train])
            y_real[ind].append(target.iloc[test])                                   # 真实结果
            y_proba[ind].append(pipeline.predict_proba(data.iloc[test])[:, 1])      # 预测概率
            f_importance.append(pipeline[type(clf).__name__].feature_importances_)  # 特征重要性
    display(y_real, y_proba, f_importance, data.columns)
