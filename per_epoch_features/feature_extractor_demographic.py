#!/usr/bin/env python
"""DemographicMixin for FeatureExtractor."""

import numpy as np
from .helper_code import load_age, load_sex, load_bmi, get_standardized_race

class DemographicMixin:
    def extract_demographic_features(self, data):
        """
        Extracts and encodes demographic features from a metadata dictionary.

        Inputs:
            data (dict): A dictionary containing patient metadata (e.g., from a CSV row).

        Returns:
            np.array: A feature vector of length 11:
                - [0]: Age (Continuous)
                - [1:4]: Sex (One-hot: Female, Male, Other/Unknown)
                - [4:9]: Race (One-hot: Asian, Black, Other, Unavailable, White)
                - [9]: BMI (Continuous)
        """
        # 1. Age Feature (1 dimension)
        # Convert 'Age' to a float; default to 0 if missing
        age = np.array([load_age(data)])

        # 2. Sex One-Hot Encoding (3 dimensions: Female, Male, Other/Unknown)
        # Uses lowercase prefix matching to handle variants like 'F', 'Female', 'M', or 'Male'
        sex = load_sex(data)
        sex_vec = np.zeros(3)
        if sex == 'Female':
            sex_vec[0] = 1 # Index 0: Female
        elif sex == 'Male':
            sex_vec[1] = 1 # Index 1: Male
        else:
            sex_vec[2] = 1 # Index 2: Other/Unknown

        # 3. Race One-Hot Encoding (6 dimensions)
        # Standardizes the raw text into one of six categories using the helper function
        race_category = get_standardized_race(data).lower()
        race_vec = np.zeros(5)
        # Pre-defined mapping for index consistency
        race_mapping = {'asian': 0, 'black': 1, 'others': 2, 'unavailable': 3, 'white': 4}
        race_vec[race_mapping.get(race_category, 2)] = 1

        # 4. Body Mass Index (BMI) Feature (1 dimension)
        # Extracts the pre-calculated mean BMI; handles strings, NaNs, and missing keys
        bmi = np.array([load_bmi(data)])

        # 5. Concatenate all components into a single vector (1 + 3 + 5 + 1 = 10)

        return np.concatenate([age, sex_vec, race_vec, bmi])
