import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            # logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")
            logging.info("StandardScaler Initialized")
            # Load schema configurations
            num_features = self._schema_config['num_features']
            # mm_columns = self._schema_config['mm_columns']

            logging.info("Cols loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),

                    # ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e
    def _replace_values_in_features(self,df):
        df.loc[:, 'PAY_0':'PAY_6'] = df.loc[:, 'PAY_0':'PAY_6'].replace(-1,0)
        fil_ed = (df['EDUCATION'] == 0) | (df['EDUCATION'] == 5) | (df['EDUCATION'] == 6)
        df.loc[fil_ed, 'EDUCATION'] = 4
        fil_mar = (df['MARRIAGE'] == 0)
        df.loc[fil_mar, 'MARRIAGE'] = 3
        logging.info("Added undefined values with 'others' category for Education,MARRIAGE and PAY's columns")
        return df
    # def _create_dummy_columns(self, df):
    #     """Create dummy variables for categorical features."""
    #     categorical_features = self._schema_config['categorical_columns']
    #     logging.info("Creating dummy variables for categorical features")
    #     df = pd.get_dummies(df, columns= categorical_features, drop_first=True)
    #     return df
    

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

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            # Apply custom transformations in specified sequence
            # input_feature_train_df = self._drop_id_column(input_feature_train_df)
            # input_feature_train_df = self._replace_values_in_features(input_feature_train_df)
            # input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
            
            # input_feature_test_df = self._drop_id_column(input_feature_test_df)
            # input_feature_test_df = self._replace_values_in_features(input_feature_test_df)
            # input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
            # Train data transformation
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_train_df = self._replace_values_in_features(input_feature_train_df)
            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
            all_columns = input_feature_train_df.columns.tolist()

            # Test data transformation
            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            input_feature_test_df = self._replace_values_in_features(input_feature_test_df)
            input_feature_test_df = self._create_dummy_columns(input_feature_test_df, all_columns=all_columns)

            
            logging.info("Custom transformations applied to train and test data")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )

            logging.info("SMOTEENN applied to train df.")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_numpy_array_data(self.data_transformation_config.all_columns_path, array = all_columns)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                all_columns_file_path=self.data_transformation_config.all_columns_path
            )

        except Exception as e:
            raise MyException(e, sys) from e