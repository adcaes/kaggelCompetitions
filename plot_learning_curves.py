"""
========================
Plotting Learning Curves
========================
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
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

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



def load_data(path):
    return pd.io.parsers.read_csv(path)


def preprocess_train(train):
    train_y = train.Survived.values

    mapper = DataFrameMapper([
        ('Pclass', preprocessing.LabelBinarizer()),
        ('Sex', preprocessing.LabelBinarizer()),
        ('Age', None),
        ('SibSp', preprocessing.Binarizer()),
        ('Parch', preprocessing.Binarizer()),
        ('Embarked', preprocessing.LabelBinarizer()),
        ('Fare', None),
    ])
    train_X = mapper.fit_transform(train)

    imputer = preprocessing.Imputer(strategy='mean')
    train_X = imputer.fit_transform(train_X)

    return train_X, train_y, mapper, imputer




train = load_data("titanic/train.csv")
X, y, mapper, imputer = preprocess_train(train)

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=5, test_size=0.2, random_state=0)


normalization = preprocessing.StandardScaler()
pca = PCA()
svc = svm.SVC(C=1, gamma=0.0316227766016837911) #mean: 0.82043, std: 0.00159, params: {'svc__gamma': 0.031622776601683791, 'pca__n_components': 9, 'svc__C': 1.0}

estimator = Pipeline(steps=[('normalization', normalization), ('pca', pca), ('svc', svc)])

title = "Learning Curves"
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.show()
