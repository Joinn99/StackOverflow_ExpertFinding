# utils.py
# 功能函数
# python3.x

import sqlite3
from time import strftime, localtime
import pandas as pd
import numpy as np
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import quantile_transform, binarize, maxabs_scale


def cur():
    """
    获取当前时间，用作随机数种子
    """
    return strftime("(%Y-%m-%d %X)", localtime())

def get_tags():
    """
    获取数据库中存储的话题标签列表
    """
    conn = sqlite3.connect('Data/StackExpert.sqlite')
    tags = list(pd.read_sql_query("""
        SELECT name FROM `sqlite_master`
        WHERE type='table' AND NOT name='sqlite_sequence'
    """, conn)['name'])
    conn.close()
    return tags

def pre_proc(data):
    """
    数据预处理函数
    """
    data = data.set_index('id')
    data.update(np.log10(data.filter(regex=r'A_SCORE$', axis=1).where(
        data.filter(regex=r'A_SCORE$', axis=1) >= 0, 0)+1))             # 取分数数据的对数
    return data.fillna(0.0)

def load_data(tag):
    """
    从数据库加载数据
    """
    # 连接数据库
    conn = sqlite3.connect('Data/StackExpert.sqlite')
    # 加载所有数据
    data = pd.read_sql_query("""
        SELECT * FROM {:s}
    """.format(tag), conn)
    conn.close()
    # 数据预处理
    data = pre_proc(data)
    # 依照专家用户的判断标准设定用户标签
    target = column_or_1d(binarize(pd.DataFrame(data.pop('EXPERT_SCORE')),
                                   threshold=99, copy=False)).astype(int)
    col = data.columns
    ind = data.index
    # 数据标准化归一化
    data = quantile_transform(data, copy=False, output_distribution='normal')
    data = pd.DataFrame(maxabs_scale(data), index=ind, columns=col)
    # 计算专家/非专家比例
    cnt = np.bincount(target)
    target = pd.Series(target, index=ind)
    return data, target, np.divide(np.min(cnt), np.max(cnt))


if __name__ == "__main__":
    pass
