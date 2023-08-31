import numpy as np
from qwak.feature_store.offline import OfflineClient
import qwak
from qwak.model.base import QwakModel
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import pandas as pd

import os

RUNNING_FILE_ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class RiskModel(QwakModel):

    def __init__(self):
        self.params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'eval_metric': 'Accuracy',
            'logging_level': 'Silent',
            'use_best_model': True
        }
        self.catboost = CatBoostClassifier(**self.params)

    def fetch_features(self):
        """
        Read data from the offline feature store
        :return: Feature Store DF
        """
        print("Fetching data from the feature store")
        offline_feature_store = OfflineClient()
        population_df = pd.read_csv(f"{RUNNING_FILE_ABSOLUTE_PATH}/population.csv")

        key_to_features = {
            'user_id': [
                'user-properties.job',
                'user-properties.credit_amount',
                'user-properties.duration',
                'user-properties.purpose',
                'user-properties.risk'
            ],
        }

        return offline_feature_store.get_feature_values(
            entity_key_to_features=key_to_features,
            population=population_df,
            point_in_time_column_name='timestamp')

    def build(self):
        """
        Build the Qwak model:
            1. Fetch the feature values from the feature store
            2. Train a naive Catboost model
        """
        df = self.fetch_features()
        train_df = df[["job", "credit_amount", "duration", "purpose"]]
        
        y = df["risk"].map({'good':1,'bad':0})


        categorical_features_indices = np.where(train_df.dtypes != np.float64)[0]
        X_train, X_validation, y_train, y_validation = train_test_split(train_df, y, test_size=0.25, random_state=42)

        train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
        validate_pool = Pool(X_validation, y_validation, cat_features=categorical_features_indices)

        print("Fitting catboost model")
        self.catboost.fit(train_pool, eval_set=validate_pool)

        y_predicted = self.catboost.predict(X_validation)
        f1 = f1_score(y_validation, y_predicted)
        qwak.log_metric({'f1_score': f1})

    def schema(self):
        from qwak.model.schema import ModelSchema, InferenceOutput, FeatureStoreInput, Entity
        user_id = Entity(name="user_id", type=str)
        model_schema = ModelSchema(
            entities=[user_id],
            inputs=[
                FeatureStoreInput(entity=user_id, name='user-properties.job'),
                FeatureStoreInput(entity=user_id, name='user-properties.credit_amount'),
                FeatureStoreInput(entity=user_id, name='user-properties.duration'),
                FeatureStoreInput(entity=user_id, name='user-properties.purpose'),

            ],
            outputs=[
                InferenceOutput(name="Risk", type=float)
            ])
        return model_schema

    @qwak.api(feature_extraction=True)
    def predict(self, df, extracted_df):
        renamed = extracted_df.rename(columns={"user-properties.job": "job","user-properties.credit_amount": "credit_amount", "user-properties.duration": "duration","user-properties.purpose": "purpose"})
        return pd.DataFrame(self.catboost.predict(renamed[["job", "credit_amount", "duration", "purpose"]]),
                            columns=['Risk'])


if __name__ == '__main__':
    model = RiskModel()
    model.build()

    feature_vector = pd.DataFrame([{
        "user_id": "e41160de-0a56-47cf-8193-a0c97fe2e752"
    }])

    print("Predicting with Feature Store!")
    print(model.predict(feature_vector, None))
