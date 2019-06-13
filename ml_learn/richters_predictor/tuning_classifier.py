import numpy as np
from sklearn.metrics import  f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from ml_learn.richters_predictor.data_utils import load_combined_data, split_data, show_feature


def predict():

    # 1 Pick Features
    feature_names = {'age': np.int, 'height_percentage': np.int, 'count_floors_pre_eq': np.int}
    label_names = {'damage_grade': np.float64}

    features_with_label = load_combined_data(feature_names, label_names)


    # 2 Remove Outliers
    features_with_label = features_with_label.query('age < 900')

    # 3 Create New features

    # TODO

    # 4 Pick a classifier

    scaler = StandardScaler()
    features_with_label.loc[:, ['age', 'height_percentage', 'count_floors_pre_eq']] \
        = scaler.fit_transform(features_with_label[['age', 'height_percentage', 'count_floors_pre_eq']])

    # load test data and take 20% as training data
    features_train, features_test, labels_train, labels_test = split_data(features_with_label, 0.2)

    # 5 Tune the classifier

    # we want to tune the hyperparameters criterion and max_depth
    criterion = ['gini', 'entropy']
    max_depth = [4, 50]

    # train directly using KFold (not the best solution)
    kf = KFold(n_splits=50)
    folds = kf.split(features_train)
    for depth in max_depth:
        train = next(folds)[0]
        feature_set = features_train.iloc[train]
        label_set = labels_train[train]
        # Here we instantiate our decision tree classifier with the current depth and train / evaluate using the current
        # fold of data
        classifier = DecisionTreeClassifier(max_depth=depth)
        classifier.fit(feature_set, label_set)
        labels_predict = classifier.predict(features_test)
        print(f'score for depth {depth}: {f1_score(labels_predict, labels_test, average="micro")}')

    # A better way to tune hyper parameters is to use GridSearchCV that will perform the cross validation, training and
    # evaluation for you
    tree_para = {'criterion': criterion,
                 'max_depth': max_depth}
    clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=5, scoring='f1_micro')
    clf.fit(features_train, labels_train)
    labels_predict = clf.predict(features_test)
    print(clf.cv_results_)
    print(f1_score(labels_predict, labels_test, average='micro'))

if __name__ == '__main__':
    predict()
