import numpy as np
import pandas as pd
from sklearn.metrics import  f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from ml_learn.richters_predictor.data_utils import load_combined_data, split_data, one_hot_encode_categories


def predict():

    # 1 Pick Features
    feature_names = {'geo_level_1_id': np.int, 'geo_level_2_id': np.int, 'geo_level_3_id': np.int,
                     'count_floors_pre_eq': np.int, 'age': np.int, 'area_percentage': np.int,
                     'height_percentage': np.int, 'land_surface_condition': np.str, 'foundation_type': np.str,
                     'roof_type': np.str, 'ground_floor_type': np.str, 'other_floor_type': np.str,
                     'position': np.str, 'plan_configuration': np.str, 'has_superstructure_adobe_mud': np.bool,
                     'has_superstructure_mud_mortar_stone': np.bool, 'has_superstructure_stone_flag': np.bool,
                     'has_superstructure_cement_mortar_stone': np.bool, 'has_superstructure_mud_mortar_brick': np.bool,
                     'has_superstructure_cement_mortar_brick': np.bool, 'has_superstructure_timber': np.bool,
                     'has_superstructure_bamboo': np.bool, 'has_superstructure_rc_non_engineered': np.bool,
                     'has_superstructure_rc_engineered': np.bool, 'has_superstructure_other': np.bool,
                     'legal_ownership_status': np.str, 'count_families': np.int, 'has_secondary_use': np.bool,
                     'has_secondary_use_agriculture': np.bool, 'has_secondary_use_hotel': np.bool,
                     'has_secondary_use_rental': np.bool, 'has_secondary_use_institution': np.bool,
                     'has_secondary_use_school': np.bool, 'has_secondary_use_industry': np.bool,
                     'has_secondary_use_health_post': np.bool, 'has_secondary_use_gov_office': np.bool,
                     'has_secondary_use_use_police': np.bool, 'has_secondary_use_other': np.bool}

    label_names = {'damage_grade': np.float64}

    features_with_label = load_combined_data(feature_names, label_names)

    categories = {
        'land_surface_condition': ['n', 'o', 't'],
        'foundation_type': ['h', 'i', 'r', 'u', 'w'],
        'roof_type': ['n', 'q', 'x'],
        'ground_floor_type': ['f', 'm', 'v', 'x', 'z'],
        'other_floor_type': ['j', 'q', 's', 'x'],
        'position': ['j', 'o', 's', 't'],
        'plan_configuration': ['a', 'c', 'd', 'f', 'm', 'n', 'o', 'q', 's', 'u'],
        'legal_ownership_status': ['a', 'r', 'v', 'w']
    }

    features_with_label = one_hot_encode_categories(categories, features_with_label)

    # 2 Remove Outliers
    features_with_label = features_with_label.query('age < 900')

    # 3 Create New features

    # TODO

    # 4 Pick a classifier

    scaler = StandardScaler()
    features_with_label.loc[:, ['age', 'height_percentage','area_percentage', 'count_floors_pre_eq', 'count_families']] \
        = scaler.fit_transform(features_with_label[['age', 'height_percentage','area_percentage', 'count_floors_pre_eq', 'count_families']])

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
