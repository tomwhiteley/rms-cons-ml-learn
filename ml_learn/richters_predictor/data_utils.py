import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as plt


def load_combined_data(feature_names, label_names):
    """
    Load data from the csv files and combine
    :param feature_names:
    :param label_names:
    :return: data frame with combined features and labels
    """
    if feature_names is None:
        train_values = pd.read_csv('../../data/train_values.csv')
    else:
        train_values = pd.read_csv('../../data/train_values.csv', usecols=feature_names.keys(), dtype=feature_names)

    if label_names is None:
        train_labels = pd.read_csv('../../data/train_labels.csv')
    else:
        train_labels = pd.read_csv('../../data/train_labels.csv', usecols=label_names.keys(), dtype=label_names)

    return train_values.merge(train_labels, left_index=True, right_index=True)


def split_data(features_with_label, test_size):
    """
    splits data into features, lables test and train
    :param features_with_label:
    :param test_size:
    :return: features_train, features_test, labels_train, labels_test
    """
    labels = features_with_label[['damage_grade']].values.ravel()
    features = features_with_label.drop(['damage_grade'], axis=1)
    return train_test_split(features, labels, test_size=test_size, random_state=123)


def show_feature(features_with_label, feature):
    """
    show historgram for a particular feature
    :param features_with_label:
    :param feature:
    :return:
    """
    g = sb.FacetGrid(features_with_label, row="damage_grade", height=2.5, aspect=2)
    g.map(sb.distplot, feature, hist=True, rug=True, kde=False, norm_hist=False)
    plt.show()

