# stackoptimizer.py
# 网格搜索法优化模型性能
# python3.x

import sqlite3
import os
import json
from time import time, sleep
from tqdm import tqdm
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from model.utils import load_data, cur, get_tags

# 模型优化器
class StackExpOptimizer():
    def __init__(self, scoring='f1'):
        """
        初始化
        """
        self.scoring = scoring      # 优化指标（默认f1-score）
        self.info = sqlite3.connect('Data/info.sqlite')
        with open('Data/params.json', 'r') as par_file:
            self.params = json.load(par_file)
        self.validator = RepeatedStratifiedKFold(n_splits=3, n_repeats=2)
        self.seed = int(time())     # 随机数种子
        self.tag = ""

    def __del__(self):
        """
        析构函数
        """
        self.info.commit()
        self.info.close()
        tqdm.write(
            "{}\n[INFO] Data stored in database. {:s}".format('*'*50, cur()))

    def _get_params(self, model, dt_ratio):
        """
        获取模型的参数
        """
        est = [i for (i, _) in model.get_params()['steps']]
        pipar = {est[1]+'__'+k: v for (k, v)
                 in self.params[est[1]]['best'].items()}
        pipar.update({est[0]+'__'+k: v for (k, v)
                      in self.params[est[0]]['best'].items()})
        pipar.update({est[0]+'__sampling_strategy':
                      dt_ratio * self.params[est[0]]['best']['sampling_strategy']})
        return pipar

    def _get_grid(self, model, dt_ratio):
        """
        获取网格搜索法所需的参数网格
        """
        est = [i for (i, _) in model.get_params()['steps']]
        grid = {est[0]+'__sampling_strategy': dt_ratio * self._gen(
            self.params[est[0]]['opti']['sampling_strategy'])}
        grid.update({est[1]+'__'+k: self._gen(v)
                     for (k, v) in self.params[est[1]]['opti'].items()})
        return grid

    def _set_params(self, para):
        """
        将参数设置为网格搜索法中寻找到的最佳参数组合
        """
        for (key, value) in para.items():
            self.params[key.split('__')[0]]['best'][key.split('__')[1]] = value

    def _gen(self, grid_par):
        """
        随机生成符合条件的参数网格
        """
        np.random.seed(self.seed)
        if grid_par['dtype'] == 'list':
            return grid_par['list']
        sc_par = {}
        while True: # 在range范围内生成一个长度大于step的子区间[start, stop]
            [sc_par['start'], sc_par['stop']] = np.sort(
                np.random.uniform(**grid_par['range'], size=2))
            if sc_par['stop'] - sc_par['start'] >= grid_par['step']:
                break
        if grid_par['scale'] == 'linear':   # 生成[start, stop]范围内的均匀分布
            grid = np.unique(np.linspace(
                num=9, dtype=grid_par['dtype'], **sc_par))
        else:
            grid = np.geomspace(num=9, dtype=grid_par['dtype'], **sc_par)
        return grid

    def _rec(self, clf_name, samp_name, res):
        """
        将实验结果存入数据库
        """
        params = {k: v for k, v in self.params[clf_name]['best'].items(
        ) if k in self.params[clf_name]['opti'].keys()}
        params.update({'tag': self.tag, 'seed': self.seed,
                       'sampling_strategy': self.params[samp_name]['best']['sampling_strategy']})
        params.update(res.rename(
            index={n: n.split('_')[-1] for n in res.index}).to_dict())
        self.info.execute("INSERT INTO {:s} {} VALUES {}".format(
            clf_name, str(tuple(params.keys())).replace('\'', ''), tuple(params.values())))

    def _check(self, clf):
        """
        检查数据库中是否存在某种分类器的实验记录，没有则创建空表
        """
        name = self.info.execute("""""").fetchall()
        if (clf, ) not in name:
            params = ['{} {}'.format(k, v['dtype'])
                      for k, v in self.params[clf]['opti'].items()]
            params += ['{} REAL'.format(v)
                       for v in self.params['GridSearchCV']["scoring"]]
            query = """CREATE TABLE IF NOT EXISTS {:s} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        seed DATETIME NOT NULL,
                        tag STRING NOT NULL,
                        sampling_strategy REAL NOT NULL,
                        {} NOT NULL)""".format(clf, " NOT NULL, ".join(params))
            self.info.execute(query)

    def random_process(self):
        """
        重复性进行实验的主函数
        """
        tag = get_tags()
        clf = [SVC, RidgeClassifier, ExtraTreesClassifier, XGBClassifier]
        try:
            while True:
                np.random.seed(int(time()))
                self.optimize(ADASYN(n_jobs=-1),
                              clf[int(np.random.randint(len(clf)))](),
                              tag[int(np.random.randint(len(tag)))])
                for sec in range(4):
                    tqdm.write(
                        "\r\b[INFO] Next will start in {:d} seconds... 'Ctrl+C' to exit.".format(
                            5-sec), end="")
                    sleep(1)
                tqdm.write("")
                self.info.commit()
        except KeyboardInterrupt:
            tqdm.write(
                "\r\b[INFO] Process end. {}".format(cur()))

    def optimize(self, samp, clf, tag):
        """
        单次实验流程
        """
        # 打印头部信息
        os.system('cls||clear')
        self.tag = tag
        tqdm.write("""{div}\n    Stack Expert Model Optimization
            {indent}--Seed: {seed}
            {indent}--Sampling Method: {samp}
            {indent}--Classification Method: {clf}
            {indent}--Tag: {tag}\n{div}
            """.format(**{'div': '*'*50, 'seed': self.seed, 'indent': '\b'*6, 'tag': self.tag,
                          'samp': type(samp).__name__, 'clf': type(clf).__name__}))
        # 加载数据
        data, target, ratio = load_data(self.tag)
        tqdm.write("[INFO] Data loads complete. Expert ratio:{:.2f}%\t{:s}".format(
            100*ratio, cur()))
        # 设置随机数种子
        self.seed = int(time())
        self.validator.random_state = self.seed
        self._check(type(clf).__name__)
        samp.set_params(**{"random_state": self.seed})
        if 'random_state' in clf.get_params().keys():
            clf.set_params(**{"random_state": self.seed})
        # 建立过采样和分类器的流水线模型
        pipeline = Pipeline(
            [(type(samp).__name__, samp), (type(clf).__name__, clf)])
        tqdm.write(
            "[INFO] Model load completed. Start grid search...\t{:s}".format(cur()))
        # 开始进行网格搜索
        for ind, (key, value) in enumerate(tqdm(self._get_grid(pipeline, ratio).items())):
            tqdm.write('-'*15 + 'Epoch {:d}'.format(ind) + '-'*15)
            # 设置默认参数
            pipeline.set_params(**self._get_params(pipeline, ratio))
            # 建立网格搜索对象
            grid_opti = GridSearchCV(estimator=pipeline,
                                     param_grid={key: value},
                                     cv=self.validator,
                                     **self.params['GridSearchCV'])
            tqdm.write(
                "[EP{:d}] Search Paramator: {:}\t{:s}".format(ind, key, cur()))
            tqdm.write("[EP{:d}] Search Grid: {:}\t{:s}".format(
                ind, str(value) + " Fitting...", cur()))
            # 拟合模型
            grid_opti.fit(data.to_numpy(), target)
            # 输出最佳参数及对应实验指标
            df_res = pd.DataFrame(grid_opti.cv_results_)
            df_res = df_res.loc[df_res['mean_test_{:s}'.format(
                self.scoring)].idxmax()]
            tqdm.write("[EP{:d}] Fit complete. Current Score: {:}\t{:s}".format(
                ind, df_res['mean_test_{:s}'.format(self.scoring)], cur()))
            tqdm.write("\r[EP{:d}] Best: {:}\t{:s}".format(
                ind, df_res['params'], cur()))
            if '{}__sampling_strategy'.format(type(samp).__name__) in df_res['params']:
                df_res['params']['{}__sampling_strategy'.format(
                    type(samp).__name__)] /= ratio
            # 更新参数
            self._set_params(df_res['params'])
            # 存储实验结果
            self._rec(type(clf).__name__, type(samp).__name__,
                      df_res.filter(regex=r'^mean_test', axis=0))
        # 完成网格搜索，一次实验结束
        tqdm.write(
            "{:s}\n[INFO] Grid search complete.{:}\t{:s}\n".format('='*50, "", cur()))
        del data, target


if __name__ == "__main__":
    SO = StackExpOptimizer("Data")
    SO.random_process()
