from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

# https://en.wikipedia.org/wiki/Naive_Bayes_classifier


person_details = pd.DataFrame({
    'Is Female': [0, 0, 0, 0, 1, 1, 1, 1],
    'Height (ft)': [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75],
    'Weight (lbs)': [180, 190, 170, 165, 100, 150, 130, 150],
    'Foot size (inches)': [12, 11, 12, 10, 6, 8, 7, 9]
})


person_details_train = person_details[['Height (ft)', 'Weight (lbs)', 'Foot size (inches)']]
labels_train = person_details['Is Female']

person_details_test = pd.DataFrame({
    'Height (ft)': [6],
    'Weight (lbs)': [130],
    'Foot size (inches)': [8]
})

expected_prediction = pd.DataFrame({
    'Is Female': [1]
})

gnb = GaussianNB()
gnb.fit(person_details_train, labels_train)
actual_prediction = gnb.predict(person_details_test)

assert np.array(expected_prediction['Is Female']) == actual_prediction
