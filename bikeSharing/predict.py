''' Prediction for Kaggle competition http://www.kaggle.com/c/bike-sharing-demand '''

import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import tree, ensemble, linear_model, svm
from sklearn.metrics import make_scorer
import numpy as np


scorer = None


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(data):
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['hour'] = map(lambda r: r.hour, data.datetime)
    data['weekday'] = map(lambda r: r.weekday(), data.datetime)
    data['month'] = map(lambda r: r.month, data.datetime)
    data['is_sunday'] = map(lambda r: r == 6, data.weekday)
    data['is_night'] = map(lambda r: 1 if 0 >= r <= 6 else 0, data.hour)
    data['bad_weather'] = map(lambda r: r > 2, data.weather)
    data['year'] = map(lambda r: r.year, data.datetime)


def preprocess_train(train):
    train_y = train['count']
    train_y1 = train['casual']
    train_y2 = train['registered']

    preprocess_data(train)

    mapper = DataFrameMapper([
        ('hour', None),
        ('season', preprocessing.LabelBinarizer()),
        ('holiday', None),
        ('workingday', None),
        ('weather', preprocessing.LabelBinarizer()),
        ('temp', None),
        ('atemp', None),
        ('humidity', None),
        ('windspeed', None),
        ('weekday', None),
        ('is_sunday', None),
        ('bad_weather', None),
        ('year', None),
    ])

    train_X = mapper.fit_transform(train)
    return train_X, train_y, train_y1, train_y2, mapper


def preprocess_test(test, mapper):
    preprocess_data(test)
    test_X = mapper.transform(test)
    return test_X


def rmsele(actual, pred):
    squared_errors = (np.log(pred + 1) - np.log(actual + 1)) ** 2
    mean_squared = np.sum(squared_errors) / len(squared_errors)
    return np.sqrt(mean_squared)


def get_rmsele_scorer():
    return make_scorer(rmsele, greater_is_better=False)


def estimate_pca(X):
    pca = PCA()
    pca.fit(X)
    return pca.explained_variance_


def create_estimator_ridge(X, y):
    pca = PCA()
    polynomial = preprocessing.PolynomialFeatures()
    normalization = preprocessing.StandardScaler()
    ridge = linear_model.Ridge()

    degree = [2]#[1, 2, 3]
    n_components = [13]#[10, 13, 15]
    alpha = [1.0]#[1.0, 10.0, 15, 20]

    pipe = Pipeline(steps=[('normalization', normalization), ('pca', pca), ('poly', polynomial), ('ridge', ridge)])
    clf = GridSearchCV(pipe,
                       dict(poly__degree=degree,
                            ridge__alpha=alpha,
                            pca__n_components=n_components),
                       n_jobs=4,
                       cv=3)

    clf.fit(X, y)
    return clf


def create_estimator_decission_tree(X, y):
    normalization = preprocessing.StandardScaler()
    dtree = tree.DecisionTreeRegressor()

    max_depth = [None, 5, 10, 15]

    pipe = Pipeline(steps=[('normalization', normalization), ('tree', dtree)])
    clf = GridSearchCV(pipe,
                       dict(tree__max_depth=max_depth),
                       n_jobs=4,
                       cv=3)

    clf.fit(X, y)
    return clf


def create_estimator_random_forest(X, y):
    normalization = preprocessing.StandardScaler()
    rforest = ensemble.RandomForestRegressor()

    n_estimators = [5, 10, 20, 25]
    max_depth = [None, 5, 10, 15]

    pipe = Pipeline(steps=[('normalization', normalization), ('rforest', rforest)])
    clf = GridSearchCV(pipe,
                       dict(rforest__n_estimators=n_estimators,
                            rforest__max_depth=max_depth),
                       n_jobs=4,
                       cv=3,
                       scoring=scorer)

    clf.fit(X, y)
    return clf


def create_estimator_gbm(X, y):
    normalization = preprocessing.StandardScaler()
    gbm = ensemble.GradientBoostingRegressor()

    n_estimators = [5, 10, 20, 25]
    max_depth = [None, 5, 10, 15]

    pipe = Pipeline(steps=[('normalization', normalization), ('gbm', gbm)])
    clf = GridSearchCV(pipe,
                       dict(gbm__n_estimators=n_estimators,
                            gbm__max_depth=max_depth),
                       n_jobs=4,
                       cv=3)

    clf.fit(X, y)
    return clf


def create_estimator_svr(X, y):
    normalization = preprocessing.StandardScaler()
    svr = svm.SVR()

    C = [0.1, 1, 10]

    pipe = Pipeline(steps=[('normalization', normalization), ('svr', svr)])
    clf = GridSearchCV(pipe,
                       dict(svr__C=C,),
                       n_jobs=4,
                       cv=3)

    clf.fit(X, y)
    return clf


def save_result(test, y, file_name):
    res = pd.DataFrame()
    res['datetime'] = test.datetime
    res['count'] = y
    res.to_csv(file_name + '.csv', index=False)


def main():
    train = load_data("train.csv")
    train_X, train_y, train_y1, train_y2, mapper = preprocess_train(train)

    global scorer
    scorer = get_rmsele_scorer()

    estimator_y1 = create_estimator_random_forest(train_X, train_y1)
    best_score_y1 = estimator_y1.best_score_

    estimator_y2 = create_estimator_random_forest(train_X, train_y2)
    best_score_y2 = estimator_y2.best_score_
    import ipdb; ipdb.set_trace()

    test = load_data("test.csv")
    test_X = preprocess_test(test, mapper)
    test_y1 = estimator_y1.predict(test_X)
    test_y2 = estimator_y2.predict(test_X)

    test_y = test_y1 + test_y2
    test_y_pos = map(lambda v: 0 if v < 0 else v, test_y)
    save_result(test, test_y_pos, "estimation_rforest_y1y2_%f_%f" % (best_score_y1, best_score_y2))


if __name__ == '__main__':
    main()
