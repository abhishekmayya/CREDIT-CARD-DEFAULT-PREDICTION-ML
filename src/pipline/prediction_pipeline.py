import sys
from src.entity.config_entity import CreditCardDefaultPredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from src.constants import SCHEMA_FILE_PATH
from pandas import DataFrame
import pandas as pd
from src.utils.main_utils import load_numpy_array_data,load_object,read_yaml_file

class CreditCardDefaultData:
    def __init__(self,
                LIMIT_BAL,
                SEX,
                EDUCATION,
                MARRIAGE,
                AGE,
                PAY_0,
                PAY_2,
                PAY_3,
                PAY_4,
                PAY_5,
                PAY_6,
                BILL_AMT1,
                BILL_AMT2,
                BILL_AMT3,
                BILL_AMT4,
                BILL_AMT5,
                BILL_AMT6,
                PAY_AMT1,
                PAY_AMT2,
                PAY_AMT3,
                PAY_AMT4,
                PAY_AMT5,
                PAY_AMT6
                ):
        try:
            self.LIMIT_BAL = LIMIT_BAL
            self.SEX = SEX
            self.EDUCATION = EDUCATION
            self.MARRIAGE = MARRIAGE
            self.AGE = AGE
            self.PAY_0 = PAY_0
            self.PAY_2 = PAY_2
            self.PAY_3 = PAY_3
            self.PAY_4 = PAY_4
            self.PAY_5 = PAY_5
            self.PAY_6 = PAY_6
            self.BILL_AMT1 = BILL_AMT1
            self.BILL_AMT2 = BILL_AMT2
            self.BILL_AMT3 = BILL_AMT3
            self.BILL_AMT4 = BILL_AMT4
            self.BILL_AMT5 = BILL_AMT5
            self.BILL_AMT6 = BILL_AMT6
            self.PAY_AMT1 = PAY_AMT1
            self.PAY_AMT2 = PAY_AMT2
            self.PAY_AMT3 = PAY_AMT3
            self.PAY_AMT4 = PAY_AMT4
            self.PAY_AMT5 = PAY_AMT5
            self.PAY_AMT6 = PAY_AMT6

        except Exception as e:
            raise MyException(e, sys) from e

    def get_input_data_frame(self) -> DataFrame:
        try:
            input_dict = self.get_data_as_dict()
            return DataFrame(input_dict)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_data_as_dict(self):
        try:
            input_data = {
                "LIMIT_BAL": [self.LIMIT_BAL],
                "SEX": [self.SEX],
                "EDUCATION": [self.EDUCATION],
                "MARRIAGE": [self.MARRIAGE],
                "AGE": [self.AGE],
                "PAY_0": [self.PAY_0],
                "PAY_2": [self.PAY_2],
                "PAY_3": [self.PAY_3],
                "PAY_4": [self.PAY_4],
                "PAY_5": [self.PAY_5],
                "PAY_6": [self.PAY_6],
                "BILL_AMT1": [self.BILL_AMT1],
                "BILL_AMT2": [self.BILL_AMT2],
                "BILL_AMT3": [self.BILL_AMT3],
                "BILL_AMT4": [self.BILL_AMT4],
                "BILL_AMT5": [self.BILL_AMT5],
                "BILL_AMT6": [self.BILL_AMT6],
                "PAY_AMT1": [self.PAY_AMT1],
                "PAY_AMT2": [self.PAY_AMT2],
                "PAY_AMT3": [self.PAY_AMT3],
                "PAY_AMT4": [self.PAY_AMT4],
                "PAY_AMT5": [self.PAY_AMT5],
                "PAY_AMT6": [self.PAY_AMT6]
            }
            return input_data
        except Exception as e:
            raise MyException(e, sys) from e


class CreditCardDefaultPredictor:
    def __init__(self,prediction_pipeline_config: CreditCardDefaultPredictorConfig = CreditCardDefaultPredictorConfig()) -> None:
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    def _replace_values_in_features(self, df):
        df.loc[:, 'PAY_0':'PAY_6'] = df.loc[:, 'PAY_0':'PAY_6'].replace(-1, 0)
        fil_ed = (df['EDUCATION'] == 0) | (df['EDUCATION'] == 5) | (df['EDUCATION'] == 6)
        df.loc[fil_ed, 'EDUCATION'] = 4
        fil_mar = (df['MARRIAGE'] == 0)
        df.loc[fil_mar, 'MARRIAGE'] = 3
        return df

    def _create_dummy_columns(self, df, all_columns=None):
        categorical_features = self._schema_config['categorical_columns']
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

        if all_columns is not None:
            missing_cols = set(all_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            df = df[all_columns]

        return df

    def _drop_id_column(self, df):
        drop_col = self._schema_config['drop_columns']
        for col in drop_col:
            if col in df.columns:
                df = df.drop(col, axis=1)
        return df

    def predict(self, dataframe) -> str:
        try:
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            logging.info("Prediction data loaded and now transforming it for prediction...")

            all_columns = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
            'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
            'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'SEX_2',
            'EDUCATION_2', 'EDUCATION_3', 'EDUCATION_4', 'MARRIAGE_2',
            'MARRIAGE_3', 'PAY_0_0', 'PAY_0_1', 'PAY_0_2', 'PAY_0_3',
            'PAY_0_4', 'PAY_0_5', 'PAY_0_6', 'PAY_0_7', 'PAY_0_8', 'PAY_2_0',
            'PAY_2_1', 'PAY_2_2', 'PAY_2_3', 'PAY_2_4', 'PAY_2_5', 'PAY_2_6',
            'PAY_2_7', 'PAY_3_0', 'PAY_3_1', 'PAY_3_2', 'PAY_3_3', 'PAY_3_4',
            'PAY_3_5', 'PAY_3_6', 'PAY_3_7', 'PAY_3_8', 'PAY_4_0', 'PAY_4_1',
            'PAY_4_2', 'PAY_4_3', 'PAY_4_4', 'PAY_4_5', 'PAY_4_6', 'PAY_4_7',
            'PAY_4_8', 'PAY_5_0', 'PAY_5_2', 'PAY_5_3', 'PAY_5_4', 'PAY_5_5',
            'PAY_5_6', 'PAY_5_7', 'PAY_6_0', 'PAY_6_2', 'PAY_6_3', 'PAY_6_4',
            'PAY_6_5', 'PAY_6_6', 'PAY_6_7', 'PAY_6_8']
            all_columns = [str(col) for col in all_columns]

            dataframe = self._drop_id_column(dataframe)
            dataframe = self._replace_values_in_features(dataframe)
            dataframe = self._create_dummy_columns(dataframe,all_columns=all_columns)
            logging.info("Tranformation completed!")
            result = model.predict(dataframe)
            logging.info("Prediction Done!")
            return result
        except Exception as e:
            raise MyException(e, sys)
