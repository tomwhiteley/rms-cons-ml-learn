import numpy as np
from sklearn.metrics import  f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from ml_learn.richters_predictor.data_utils import load_combined_data, split_data, show_feature


def predict():

    # 1 Pick Features
    feature_names = {'age': np.int, 'height_percentage': np.int, 'count_floors_pre_eq': np.int}
    label_names = {'damage_grade': np.float64}

    features_with_label = load_combined_data(feature_names, label_names)

    # visualise features
    show_feature(features_with_label, 'age')
    show_feature(features_with_label, 'height_percentage')
    show_feature(features_with_label, 'count_floors_pre_eq')

    # 2 Remove Outliers
    features_with_label = features_with_label.query('age < 900')

    # 3 Create New features

    # TODO

    # 4 Pick a classifier

    scaler = MinMaxScaler()
    features_with_label[['age', 'height_percentage', 'count_floors_pre_eq']] \
        = scaler.fit_transform(features_with_label[['age', 'height_percentage', 'count_floors_pre_eq']])

    # load test data and take 20% as training data
    features_train, features_test, labels_train, labels_test = split_data(features_with_label, 0.2)

    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    labels_predict = clf.predict(features_test)

    print(f1_score(labels_predict, labels_test, average='micro'))

    # 5 Tune the classifier

    # TODO


if __name__ == '__main__':
    predict()
