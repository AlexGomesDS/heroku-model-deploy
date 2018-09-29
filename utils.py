import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders

class DataCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, n_min=30):
        self.other_factors_cols = ["other_factor_1",
                                   "other_factor_2",
                                   "other_factor_3"]
        self.n_min = n_min

    def transform(self, X):
        df_other_factors = X[self.other_factors_cols].copy().fillna("N/A or Unknown")

        ohe = np.zeros([X.shape[0], len(self.factors)])
        for key, row in X.iterrows():
            for el in row:
                if el in self.factors:
                    ohe[key][self.factors.index(el)] = 1

        df_factors_ohe = pd.DataFrame(ohe, columns=self.factors, dtype=np.int32)

        df_X_clean = pd.concat([X, df_factors_ohe], axis=1).drop(columns=self.other_factors_cols)

        # convert all columns to int
        df_X_clean["male"] = df_X_clean["m_or_f"].apply(self.convert_m_or_f)

        # m_or_f was ambiguous, NaNs are unnecessary
        df_X_clean = df_X_clean.drop(columns=["m_or_f", "N/A or Unknown"])

        df_X_clean = self.enc.transform(df_X_clean)

        # drop features that only appeared n_min or less times in the trainning set
        X_transformed = df_X_clean[self.usable_cols].copy()

        return X_transformed

    def fit(self, df_x, y):
        df_other_factors = df_x[self.other_factors_cols].copy().fillna("N/A or Unknown")

        # list of possible factors
        self.factors = list(np.unique(df_other_factors.values))

        ohe = np.zeros([df_x.shape[0], len(self.factors)])
        for key, row in df_other_factors.iterrows():
            for el in row:
                ohe[key][self.factors.index(el)] = 1

        df_factors_ohe = pd.DataFrame(ohe, columns=self.factors, dtype=np.int32)
        # drop factors with few observations
        self.factors = list(df_factors_ohe.columns[(df_factors_ohe.sum() > self.n_min)].values)
        df_factors_ohe = df_factors_ohe[self.factors]

        df_X_clean = pd.concat([df_x, df_factors_ohe], axis=1).drop(columns=self.other_factors_cols)

        # convert all columns to int
        df_X_clean["male"] = df_X_clean["m_or_f"].apply(self.convert_m_or_f)

        # m_or_f was ambiguous
        df_X_clean = df_X_clean.drop(columns=["m_or_f", "N/A or Unknown"])
        self.enc = category_encoders.one_hot.OneHotEncoder(handle_unknown='ignore')
        df_X_clean = self.enc.fit_transform(df_X_clean)

        # drop features that only appear 3 or less times
        self.usable_cols = df_X_clean.columns[df_X_clean.sum() > self.n_min].values
        self.X_train = df_X_clean[self.usable_cols].copy()
        self.y_train = y.copy()
        return self

    def convert_m_or_f(self, val):
        if val == 'm':
            return 1
        elif val == 'f':
            return 0
        else:
            return np.nan
