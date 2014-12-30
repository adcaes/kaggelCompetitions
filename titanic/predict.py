''' Code for Kaggle competition https://www.kaggle.com/c/titanic-gettingStarted '''

import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Titles with more than 10 samples in the train set
titles = {'Mr.', 'Miss.', 'Mrs.', 'Master.', 'Dr.', 'Rev.'}


def load_data(path):
    return pd.io.parsers.read_csv(path)


def get_age_by_class(data):
    return [(cl, np.median((data.Age[(data.Pclass == cl) & (data.Age.notnull())]))) for cl in [1, 2, 3]]


def get_age_by_title(data):
    return [(title, np.median((data.Age[(data.Title == title) & (data.Age.notnull())]))) for title in titles]


def set_missing_age_by_class(data, age_by_class):
    for cl, age in age_by_class:
        data.Age[(data.Pclass == cl) & (data.Age.isnull())] = age


def set_missing_age_by_title(data, age_by_title):
    for title, age in age_by_title:
        data.Age[(data.Title == title) & (data.Age.isnull())] = age


def set_title(data):
    ''' Extract passanger title from name (Rev, Dr, Mr etc) '''
    def get_title(name):
        for word in name.split():
            if word in titles:
                return word
        return 'None'

    data['Title'] = map(get_title, data.Name)


def set_fare_per_person(data):
    data['FarePerPerson'] = data.Fare/data.FamilySize


def set_family_size(data):
    data['FamilySize'] = data.SibSp + data.Pclass


def set_age_class(data):
    data['AgeClass'] = data.Age + data.Pclass


def set_sex_class(data):
    data['SexClass'] = ((data.Sex == 'female') + 1) * data.Pclass # Sex encoded as 1 or 2


def preprocess_data(data, age_by_col=None):
    set_title(data)
    set_family_size(data)
    set_fare_per_person(data)
    set_sex_class(data)

    if not age_by_col:
        age_by_col = get_age_by_title(data)

    set_missing_age_by_title(data, age_by_col)
    set_age_class(data)
    return age_by_col


def preprocess_train(train):
    train_y = train.Survived.values

    age_by_col = preprocess_data(train)
    mapper = DataFrameMapper([
        ('Pclass', preprocessing.LabelBinarizer()),
        ('Sex', preprocessing.LabelBinarizer()),
        ('Age', None),
        ('SibSp', None),
        ('Parch', None),
        ('Embarked', preprocessing.LabelBinarizer()),
        ('Fare', None),
        ('FamilySize', None),
        ('Title', preprocessing.LabelBinarizer()),
        ('FarePerPerson', None),
        ('AgeClass', None),
        ('SexClass', None)
    ])

    train_X = mapper.fit_transform(train)
    imputer = preprocessing.Imputer(strategy='mean')
    train_X = imputer.fit_transform(train_X)

    return train_X, train_y, mapper, imputer, age_by_col


def preprocess_test(test, mapper, imputer, age_by_col):
    preprocess_data(test, age_by_col)
    test_X = mapper.transform(test)
    test_X = imputer.transform(test_X)
    return test_X


def save_result(test, y, file_name):
    y_test_labelled = np.c_[test.PassengerId.values, y]
    y_test_labelled = y_test_labelled.astype(int)
    np.savetxt(file_name + '.csv', y_test_labelled, delimiter=",", header="PassengerId,Survived", fmt="%d") # Need to delete # for column labels


def plot_data(X, y):
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.fit_transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:,1], c=y)
    plt.show()


def estimate_pca(X):
    pca = PCA()
    pca.fit(X)
    return pca.explained_variance_


def create_estimator_svc(X, y):
    #pca = PCA()
    normalization = preprocessing.StandardScaler()
    svc = svm.SVC(cache_size=500)

    n_components = [16]
    c_range = 10.0 ** np.arange(-1, 5, 1)
    gamma_range = 10.0 ** np.arange(-2, 0.5, .5)

    pipe = Pipeline(steps=[('normalization', normalization), ('svc', svc)])
    clf = GridSearchCV(pipe,
                       dict(svc__C=c_range,
                            svc__gamma=gamma_range),
                       n_jobs=4,
                       cv=10)

    clf.fit(X, y) #mean: 0.836 (Kaggle 0.80861)
    return clf


def create_estimator_forest(X, y):
    pca = PCA()
    normalization = preprocessing.StandardScaler()

    rand_f = RandomForestClassifier()

    param_grid = {'n_components': [10],
                  'n_estimators': [10, 50, 100, 200, 300],
                  'criterion': ['gini', 'entropy'],
                  'max_features': ['sqrt', None, 'log2'],
                  'bootstrap': [True, False]}


    pipe = Pipeline(steps=[('pca', pca), ('normalization', normalization), ('rand_f', rand_f)])
    clf = GridSearchCV(pipe,
                       dict(rand_f__n_estimators=param_grid['n_estimators']),
                       n_jobs=4,
                       cv=10)

    clf.fit(X, y) # best 0.815
    return clf


def main():
    train = load_data("train.csv")
    train_X, train_y, mapper, imputer, age_by_col = preprocess_train(train)

    estimator = create_estimator_svc(train_X, train_y)
    best_score = estimator.best_score_

    test = load_data("test.csv")
    test_X = preprocess_test(test, mapper, imputer, age_by_col)
    test_y = estimator.predict(test_X)
    save_result(test, test_y, "estimation_svc_" + str(best_score))


if __name__ == '__main__':
    main()
