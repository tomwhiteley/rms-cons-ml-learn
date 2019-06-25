import numpy as np
import pandas as pd
from sklearn.metrics import  f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from ml_learn.richters_predictor.data_utils import load_combined_data, split_data, one_hot_encode_categories


def predict():

    # 1 Pick Features
    feature_names = {'age': np.int, 'height_percentage': np.int, 'count_floors_pre_eq': np.int,
                     'land_surface_condition': np.str, 'foundation_type': np.str, 'roof_type': np.str,
                     'ground_floor_type': np.str, 'other_floor_type': np.str, 'position': np.str,
                     'plan_configuration': np.str}
    label_names = {'damage_grade': np.float64}

    features_with_label = load_combined_data(feature_names, label_names)

    encoder = OneHotEncoder(handle_unknown="error", categories=[['n','o','t']])
    encoded_features = encoder.fit_transform(features_with_label[['land_surface_condition']])

    new_column_data = pd.DataFrame(encoded_features.toarray(), columns=['land_surface_condition_n', 'land_surface_condition_o', 'land_surface_condition_t'])

    new_features_with_label = features_with_label.drop(columns='land_surface_condition')
    new_features_with_label = pd.concat([new_features_with_label, new_column_data], axis=1)

    categories = {
        'land_surface_condition': ['n', 'o', 't'],
        'foundation_type': ['h', 'i', 'r', 'u', 'w'],
        'roof_type': ['n', 'q', 'x'],
        'ground_floor_type': ['f', 'm', 'v', 'x', 'z'],
        'other_floor_type': ['j', 'q', 's', 'x'],
        'position': ['j', 'o', 's', 't'],
        'plan_configuration': ['a', 'c', 'd', 'f', 'm', 'n', 'o', 'q', 's', 'u']
    }

    features_with_label = one_hot_encode_categories(categories, features_with_label)

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

    tree_para = {'criterion': criterion,
                 'max_depth': max_depth}
    clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=5, scoring='f1_micro')
    clf.fit(features_train, labels_train)
    labels_predict = clf.predict(features_test)
    print(clf.cv_results_)
    print(f1_score(labels_predict, labels_test, average='micro'))


if __name__ == '__main__':
    predict()
