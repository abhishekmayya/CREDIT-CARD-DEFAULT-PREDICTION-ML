from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact, DataTransformationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN,SCHEMA_FILE_PATH
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data,load_object,read_yaml_file
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 data_tranformation_artifact: DataTransformationArtifact, model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_tranformation_artifact = data_tranformation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys)
        
    def _replace_values_in_features(self,df):
        df.loc[:, 'PAY_0':'PAY_6'] = df.loc[:, 'PAY_0':'PAY_6'].replace(-1,0)
        fil_ed = (df['EDUCATION'] == 0) | (df['EDUCATION'] == 5) | (df['EDUCATION'] == 6)
        df.loc[fil_ed, 'EDUCATION'] = 4
        fil_mar = (df['MARRIAGE'] == 0)
        df.loc[fil_mar, 'MARRIAGE'] = 3
        logging.info("Added undefined values with 'others' category for Education,MARRIAGE and PAY's columns")
        return df
    

    def _create_dummy_columns(self, df, all_columns=None):
        """Create dummy variables for categorical features."""
        categorical_features = self._schema_config['categorical_columns']
        logging.info("Creating dummy variables for categorical features")
        
        # Create dummy variables
        df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
        
        # Ensure consistency between train and test columns
        if all_columns is not None:
            missing_cols = set(all_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            df = df[all_columns]  # Reorder columns to match the training set
        
        return df

    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping id columns")
        drop_col = self._schema_config['drop_columns']
        for col in drop_col:
            if col in df.columns:
                df = df.drop(col, axis=1)
        return df

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")
            all_columns = load_numpy_array_data(file_path=self.data_tranformation_artifact.all_columns_file_path)
            # all_columns = all_columns.astype(str)
            all_columns = [str(col) for col in all_columns]
            # print("columns for all columns")
            # for col in all_columns:
            #     print(f"Column: {col}, Type: {type(col)}")
            x = self._drop_id_column(x)
            x = self._replace_values_in_features(x)
            x = self._create_dummy_columns(x,all_columns=all_columns)
            # print(x.columns)
            # Check for problematic column names
            # print("columns for x")
            # for col in x.columns:
            #     print(f"Column: {col}, Type: {type(col)}")

            # x.columns = x.columns.astype(str)
            # print(x.columns)
            preprocessor_object = load_object(file_path=self.data_tranformation_artifact.transformed_object_file_path)
            logging.info("Preprocessor object loaded/exists.")
            x = preprocessor_object.transform(x)
            
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")

            best_model_f1_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e