# Stack Expert Experiment
import sqlite3
import os
import json
from time import time, sleep, strftime, localtime
from tqdm import tqdm
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import quantile_transform, binarize, maxabs_scale
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN


def cur():
    """
    Return local time as random seed.
    """
    return strftime("(%Y-%m-%d %X)", localtime())


def pre_proc(data):
    """
    Preprocess input data.
    """
    del data['id']
    data.update(np.log10(data.filter(regex=r'A_SCORE$', axis=1).where(
        data.filter(regex=r'A_SCORE$', axis=1) >= 0, 0)+1))
    return data.fillna(0.0)


def load(tag, path):
    """
    Load data from database.
    """
    # Connect database and execute query.
    conn = sqlite3.connect('{}/StackExpert.db'.format(path))
    data = pd.read_sql_query("""
        SELECT * FROM {:s} WHERE CNT_A>2
    """.format(tag), conn)
    conn.close()
    data = pre_proc(data)
    # Binarize target and normarlize data features
    target = column_or_1d(binarize(pd.DataFrame(data.pop('EXPERT_SCORE')),
                                   threshold=99, copy=False)).astype(int)
    data = quantile_transform(data, copy=False, output_distribution='normal')
    # Calculate expert/non-expert ratio.
    cnt = np.bincount(target)
    return maxabs_scale(data), target, np.divide(np.min(cnt), np.max(cnt))


class StackExpOptimizer():
    def __init__(self, path, scoring='f1'):
        self.path = path
        self.scoring = scoring
        self.info = sqlite3.connect('{:s}/info.db'.format(path))
        with open('{:s}/params.json'.format(self.path), 'r') as par_file:
            self.params = json.load(par_file)
        self.validator = RepeatedStratifiedKFold(n_splits=3, n_repeats=2)
        self.seed = int(time())
        self.tag = ""

    def __del__(self):
        self.info.commit()
        self.info.close()
        tqdm.write(
            "{}\n[INFO] Data stored in database. {:s}".format('*'*50, cur()))

    def _get_params(self, model, dt_ratio):
        """
        Get default parameterss for experiment model.
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
        Get model parameter grid for grid search.
        """
        est = [i for (i, _) in model.get_params()['steps']]
        grid = {est[0]+'__sampling_strategy': dt_ratio * self._gen(
            self.params[est[0]]['opti']['sampling_strategy'])}
        grid.update({est[1]+'__'+k: self._gen(v)
                     for (k, v) in self.params[est[1]]['opti'].items()})
        return grid

    def _set_params(self, para):
        """
        Set the best parameters of the model which find in grid search.
        """
        for (key, value) in para.items():
            self.params[key.split('__')[0]]['best'][key.split('__')[1]] = value

    def _gen(self, grid_par):
        """
        Generate random parameter grid space.
        """
        np.random.seed(self.seed)
        if grid_par['dtype'] == 'list':
            return grid_par['list']
        sc_par = {}
        [sc_par['start'], sc_par['stop']] = np.sort(
            np.random.uniform(**grid_par['range'], size=2))
        if grid_par['scale'] == 'linear':
            grid = np.unique(np.linspace(
                num=3, dtype=grid_par['dtype'], **sc_par))
        else:
            grid = np.geomspace(num=3, dtype=grid_par['dtype'], **sc_par)
        return grid

    def _rec(self, clf_name, samp_name, res):
        """
        Record experiment results.
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
        Check database. Create table if not exists.
        """
        name = self.info.execute("""SELECT name FROM `sqlite_master`
                WHERE type='table' AND NOT name='sqlite_sequence'""").fetchall()
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
        Ramdom run the experiment.
        """
        tag = ['css', 'javascript', 'java', 'python', 'c_sharp', 'php',
               'android', 'jquery', 'c_plus', 'html']
        clf = [SVC, RidgeClassifier, ExtraTreesClassifier, XGBClassifier]
        try:
            while True:
                seed = int(time())
                self.optimize(ADASYN(n_jobs=-1),
                              clf[seed % len(clf)](), tag[seed % len(tag)])
                for sec in range(5):
                    tqdm.write(
                        "\r\b[INFO] Next will start in {:d} seconds... 'Ctrl+C' to exit.".format(
                            5-sec), end="")
                    sleep(1)
                tqdm.write("")
        except KeyboardInterrupt:
            tqdm.write(
                "\r\b[INFO] Process end. {}".format(cur()))

    def optimize(self, samp, clf, tag):
        """
        Main optimize process.
        """
        # Print title and information.
        os.system('cls||clear')
        self.tag = tag
        tqdm.write("""{div}\n    Stack Expert Model Optimization
            {indent}--Seed: {seed}
            {indent}--Sampling Method: {samp}
            {indent}--Classification Method: {clf}
            {indent}--Tag: {tag}\n{div}
            """.format(**{'div': '*'*50, 'seed': self.seed, 'indent': '\b'*6, 'tag': self.tag,
                          'samp': type(samp).__name__, 'clf': type(clf).__name__}))
        # Load data.
        data, target, ratio = load(self.tag, self.path)
        tqdm.write("[INFO] Load complete. Expert ratio:{:.2f}%\t{:s}".format(
            100*ratio, cur()))
        # Set random seed.
        self.seed = int(time())
        self.validator.random_state = self.seed
        self._check(type(clf).__name__)
        samp.set_params(**{"random_state": self.seed})
        if 'random_state' in clf.get_params().keys():
            clf.set_params(**{"random_state": self.seed})
        # Construct over-sampling and classifier model.
        pipeline = Pipeline(
            [(type(samp).__name__, samp), (type(clf).__name__, clf)])
        tqdm.write(
            "[INFO] Model load completed. Start grid search...\t{:s}".format(cur()))
        # Perform grid-search with each parameters.
        for ind, (key, value) in enumerate(tqdm(self._get_grid(pipeline, ratio).items())):
            tqdm.write('-'*15 + 'Epoch {:d}'.format(ind) + '-'*15)
            # Set model default params.
            pipeline.set_params(**self._get_params(pipeline, ratio))
            # Construct grid-search object.
            grid_opti = GridSearchCV(estimator=pipeline,
                                     param_grid={key: value},
                                     cv=self.validator,
                                     **self.params['GridSearchCV'])
            tqdm.write(
                "[EP{:d}] Search Paramator: {:}\t{:s}".format(ind, key, cur()))
            tqdm.write("[EP{:d}] Search Grid: {:}\t{:s}".format(
                ind, str(value) + " Fitting...", cur()))
            # Fit the model.
            grid_opti.fit(data, target)
            # Find the best score and paramaters.
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
            # Update parameters.
            self._set_params(df_res['params'])
            # Store the results.
            self._rec(type(clf).__name__, type(samp).__name__,
                      df_res.filter(regex=r'^mean_test', axis=0))
        tqdm.write(
            "{:s}\n[INFO] Grid search complete.{:}\t{:s}\n".format('='*50, "", cur()))
        del data, target


if __name__ == "__main__":
    SEO = StackExpOptimizer(path='Data/4_Features')
    SEO.random_process()
