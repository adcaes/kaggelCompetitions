''' Code for Kaggle competition https://www.kaggle.com/c/data-science-london-scikit-learn '''

import numpy as np
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_data(X, y):
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.fit_transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:,1], c=y)
    plt.show()


def plot_error_by_param(clf, param):
    pass


def load_data():
    X = np.loadtxt("train.csv", delimiter=",")
    y = np.loadtxt("trainLabels.csv", delimiter=",")
    X_test = np.loadtxt("test.csv", delimiter=",")
    return X, y, X_test


def normalize(X, X_test):
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)
    return X, X_test


def save_result(y_test, file_name):
    y_test_labelled = np.c_[[i for i in range(1, 9001)], y_test]
    y_test_labelled = y_test_labelled.astype(int)
    np.savetxt(file_name + '.csv', y_test_labelled, delimiter=",", header="Id,Solution", fmt="%d") #Need to delete # for column labels


def create_estimator(X, y):
    pca = PCA()
    svc = svm.SVC()

    n_components = [12]
    c_range = 10.0 ** np.arange(6.5,7.5,.25)
    gamma_range = 10.0 ** np.arange(-1.5,0.5,.25)

    pipe = Pipeline(steps=[('pca', pca), ('svc', svc)])
    clf = GridSearchCV(pipe,
                       dict(pca__n_components=n_components,
                            pca__whiten=[True],
                            svc__C=c_range,
                            svc__gamma=gamma_range))

    clf.fit(X, y)
    return clf


def main():
    X, y, X_test = load_data()

    # Normalization gets worse results
    # X, X_test = normalize(X, X_test)

    estimator = create_estimator(X, y)

    best_score = estimator.best_score_ #  0.943
    y_test = estimator.predict(X_test)
    save_result(y_test, 'estimation_svc_' + str(best_score))


if __name__ == '__main__':
    main()
