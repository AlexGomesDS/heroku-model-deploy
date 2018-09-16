import os
import json
import pickle
from sklearn.externals import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict



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



########################################
# Begin database stuff

if 'DATABASE_URL' in os.environ:
    db_url = os.environ['DATABASE_URL']
    dbname = db_url.split('@')[1].split('/')[1]
    user = db_url.split('@')[0].split(':')[1].lstrip('//')
    password = db_url.split('@')[0].split(':')[2]
    host = db_url.split('@')[1].split('/')[0].split(':')[0]
    port = db_url.split('@')[1].split('/')[0].split(':')[1]
    DB = PostgresqlDatabase(
        dbname,
        user=user,
        password=password,
        host=host,
        port=port,
    )
else:
    DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.drop_tables([Prediction], safe=True)
DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json
    obs_dict = request.get_json()
    _id = obs_dict['id']
    observation = obs_dict['observation']
    # now do what we already learned in the notebooks about how to transform
    # a single observation into a dataframe that will work with a pipeline
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    # now get ourselves an actual prediction of the positive class
    proba = pipeline.predict_proba(obs)[0, 1]
    response = {'proba': proba}
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(debug=True, port=5000)
