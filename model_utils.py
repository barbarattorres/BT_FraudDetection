# ---------------------- Execute save dataframes function & print results ---------------------- #
# ---------------------- Standard Library Imports ---------------------- #
import contextlib
import logging
import os
import sys
import warnings
from collections import defaultdict
from dateutil.parser import parse
from dateutil import parser as date_parser
from typing import Dict, List, Tuple, Union

# ---------------------- Data Handling and Processing ---------------------- #
import numpy as np
import pandas as pd

# ---------------------- Data Visualization ---------------------- #
import matplotlib.pyplot as plt

# ---------------------- General Machine Learning Utilities ---------------------- #
import shap  # Model interpretability

# ---------------------- Machine Learning Models and Algorithms ---------------------- #
import lightgbm as lgb
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.cluster import DBSCAN
from sklearn.utils import class_weight
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ---------------------- Data Preprocessing and Feature Engineering ---------------------- #
from category_encoders import BinaryEncoder, TargetEncoder
from imblearn.over_sampling import SMOTE, ADASYN

# ---------------------- Model Training and Evaluation ---------------------- #
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, make_scorer, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, KFold, RepeatedStratifiedKFold, cross_val_score, train_test_split
)

# ---------------------- Neural Network Specifics ---------------------- #
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers, initializers, optimizers
from keras.layers import Dropout, BatchNormalization, LeakyReLU
from keras.callbacks import ReduceLROnPlateau

# Set TensorFlow log level to suppress messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Redirect TensorFlow logs to /dev/null
tf.get_logger().setLevel('ERROR')
tf.get_logger().handlers = [logging.NullHandler()]

# ---------------------- Hyperparameter Optimization ---------------------- #
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import HyperbandPruner, MedianPruner, PatientPruner

# ---------------------- IMPORT MY_MODULES  ---------------------- #
import common_utils as cu
import plotting_utils as pu

# ---------------------- Configuration Settings ---------------------- #
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter('ignore')

# Remove all handlers associated with the root logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up basic configuration for logging outputs
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

# Create a logger
logger = logging.getLogger()

# Disable GPU usage for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set display options for Pandas
pd.set_option('display.width', 1000)


# ====================== MODEL RANDOM FOREST: Feature Importances & Selection =======================#

def feature_importances_rf(
    df: pd.DataFrame,
    target_column: str,
    exclude_cols: List[str],
    threshold_cut: Union[None, float, str] = None,
    use_class_weight: bool = False,
    smote_sampling_strategy: Union[None, float, str] = None,
    top_n: int = 60,
    skf: StratifiedKFold = None
) -> pd.DataFrame:
    """
    Feature Importance Evaluation Using Random Forest and SHAP
    ==========================================================
    
    This function evaluates feature importance using a Random Forest classifier and SHAP values, selecting top features 
    based on this analysis. It integrates handling class imbalance via SMOTE and allows for weighting classes 
    inversely proportional to their frequencies. The resulting DataFrame is limited to selected features 
    determined by importance scores and a specified threshold.

    Parameters:
    -----------
    - df (pd.DataFrame): Dataset for analysis.
    - target_column (str): Assuming 1 is the target class for binary classification.
    - exclude_cols (List[str]): Columns to exclude from analysis.
    - threshold_cut (Union[None, float, str], optional): Threshold for selecting features based on their importance. 
      Features with an importance score below this threshold will be excluded.
    - use_class_weight (bool): If True, adjust class weights inversely proportional to their frequencies.
    - smote_sampling_strategy (Union[None, float, str], optional): Strategy for applying SMOTE to address class imbalance.
    - top_n (int): Number of top features to select based on importance. Default is 60. 
    - skf (StratifiedKFold, optional): StratifiedKFold instance for splitting the dataset. If None, a default 
      StratifiedKFold configuration must be provided by the user.

    Returns:
    --------
    - pd.DataFrame: DataFrame limited to selected features based on the threshold and top N selection.

    Raises:
    -------
    - ValueError: If any of the columns specified in `exclude_cols` are not found in the DataFrame.
    - KeyError: If the `target_column` is not found in the DataFrame.
    - RuntimeError: If an error occurs during the training of the RandomForest model or during the calculation of SHAP values.

    Note:
    -----
    - This function assumes that the input DataFrame has already been cleaned and preprocessed.
    - The `skf` parameter, a StratifiedKFold instance, must be provided. If `skf` is not provided, the function will not work.

    Usage:
    ------
    >>> from sklearn.model_selection import StratifiedKFold
    >>> df = pd.read_csv("your_dataset.csv")
    >>> skf = StratifiedKFold(n_splits=5)
    >>> selected_features_df = feature_importances_rf(
            df=df,
            target_column='target',
            exclude_cols=['id'],
            threshold_cut=0.01,
            use_class_weight=True,
            smote_sampling_strategy=0.5,
            top_n=50,
            skf=skf
        )
    >>> print(selected_features_df)

    Detailed Explanation:
    ---------------------
    1. **Initialization and Checks**: 
       - Verify if the target column exists in the DataFrame.
       - Check if the columns to be excluded exist in the DataFrame.
       - Initialize feature importances and prepare for the cross-validation loop.

    2. **Cross-Validation and Random Forest Training**: 
       - Perform Stratified K-Fold cross-validation.
       - Apply SMOTE for handling class imbalance if specified.
       - Train the Random Forest classifier and collect feature importances.

    3. **Normalization and SHAP Analysis**: 
       - Normalize the feature importances.
       - Use SHAP to calculate mean absolute SHAP values for the best Random Forest model.

    4. **Feature Selection and DataFrame Preparation**: 
       - Merge SHAP values with feature importances.
       - Select the top N features and those above the threshold cut.
       - Create a DataFrame with the selected features.

    Example:
    --------
    ```python
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    # Sample DataFrame
    df = pd.DataFrame({
        'feature1': [0.1, 0.2, 0.3, 0.4],
        'feature2': [1, 2, 3, 4],
        'target': [0, 1, 0, 1]
    })

    # Create StratifiedKFold instance
    skf = StratifiedKFold(n_splits=2)

    # Use the function
    selected_features_df = feature_importances_rf(
        df=df,
        target_column='target',
        exclude_cols=[],
        threshold_cut=0.01,
        use_class_weight=True,
        smote_sampling_strategy=0.5,
        top_n=2,
        skf=skf
    )
    ```
    """
    params = {
    'n_estimators': 100,
    'random_state': 42,
    'n_jobs': -1,
    }

    try:
        if skf is None:
            logging.error("StratifiedKFold (skf) is not provided. Please pass a valid StratifiedKFold instance.")
            return None

        # Check if target_column exists in the DataFrame
        if target_column not in df.columns:
            raise KeyError(f"The target column '{target_column}' was not found in the DataFrame.")

        # Check if any exclude column does not exist in DataFrame
        missing_cols = [col for col in exclude_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following columns to exclude are not in the DataFrame: {', '.join(missing_cols)}")

        df = df.reset_index(drop=True)
        X = df.drop(columns=[target_column] + exclude_cols)
        y = df[target_column]

        feature_importances = defaultdict(float)

        best_score = float('-inf')
        best_model = None
        best_X_val, best_y_val = None, None
        
        # Start Cross Validation
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx] 

            # Handle class imbalance with SMOTE if requested
            if smote_sampling_strategy and 0 < smote_sampling_strategy < 1:
                smote = SMOTE(random_state=42, sampling_strategy=smote_sampling_strategy)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            # Inside the cross-validation loop, before training the model
            params_copy = params.copy()

            # Dynamic use_class_weight adjustment
            if use_class_weight:
                params_copy['class_weight'] = 'balanced'

            model = RandomForestClassifier(**params_copy, oob_score=True)
            model.fit(X_train, y_train)

            # Get feature importance and update dictionary
            importances = model.feature_importances_
            for i, col in enumerate(X_train.columns):
                feature_importances[col] += importances[i] / skf.n_splits

            # Track the best model based on evaluation metric
            current_score = model.oob_score_
            if current_score > best_score:
                best_score = current_score
                best_model, best_X_val, best_y_val = model, X_val, y_val

        # Normalize feature importances
        total_importance = sum(feature_importances.values())
        normalized_importances = {k: v / total_importance for k, v in feature_importances.items()}

        # Convert to DataFrame for easier manipulation
        feature_importances_df = pd.DataFrame(list(normalized_importances.items()), 
                                                columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

        # SHAP analysis
        explainer = shap.TreeExplainer(best_model)
        shap_sample = shap.sample(best_X_val, 100, random_state=42) if len(best_X_val) > 100 else best_X_val
        shap_values = explainer.shap_values(shap_sample, check_additivity=False)

        # Select SHAP values for the positive class if it is a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Handling of SHAP values
        mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

        # Ensure SHAP values are 1-dimensional and lengths match
        if mean_abs_shap_values.shape[0] != len(best_X_val.columns):
            raise ValueError("The length of mean_abs_shap_values does not match the number of features.")

        mean_shap_df = pd.DataFrame({
            'Feature': best_X_val.columns, 
            'SHAP_Value': mean_abs_shap_values
        }).sort_values(by='SHAP_Value', ascending=False)
    
        comparison_df = mean_shap_df.merge(feature_importances_df, on='Feature', how='outer').fillna(0)
        if top_n:
            comparison_df = comparison_df.head(top_n)
        
        comparison_df = comparison_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        comparison_df['SHAP_Value'] = comparison_df['SHAP_Value'].round(4)
        comparison_df['Importance'] = comparison_df['Importance'].round(4)

        nonzero_features_df = comparison_df[comparison_df['Importance'] >= (threshold_cut if threshold_cut else 0)]
        nonzero_feature_names = nonzero_features_df['Feature'].tolist()
        
        df_selected_top_n = df[[*nonzero_feature_names, target_column] + exclude_cols]

        print('DataFrame before: ' + str(df.shape))
        print('DataFrame after: ' + str(df_selected_top_n.shape))
        print('\n')
        print(comparison_df)
        return df_selected_top_n

    except Exception as e:
        logging.error(f"An error occurred during the feature importance analysis: {e}")
        raise



# ====================== MODEL LOGREG: Feature Importances & Selection =======================#

def feature_importances_logreg(
    df: pd.DataFrame,
    target_column: str,
    exclude_cols: List[str],
    threshold_cut: Union[None, float, str] = None,
    use_class_weight: bool = False,
    smote_sampling_strategy: Union[None, float, str] = None,
    top_n: int = 60,
    skf: StratifiedKFold = None
) -> pd.DataFrame:
    """
    Feature Importance Evaluation Using Logistic Regression and SHAP
    ==========================================================
    
    This function evaluates feature importance using Logistic Regression and SHAP values, selecting top features 
    based on this analysis. It integrates handling class imbalance via SMOTE and allows for weighting classes 
    inversely proportional to their frequencies. The resulting DataFrame is limited to selected features 
    determined by importance scores and a specified threshold.

    Parameters:
    -----------
    - df (pd.DataFrame): Dataset for analysis.
    - target_column (str): Assuming 1 is the target class for binary classification.
    - exclude_cols (List[str]): Columns to exclude from analysis.
    - threshold_cut (Union[None, float, str], optional): Threshold for selecting features based on their importance. 
      Features with an importance score below this threshold will be excluded.
    - use_class_weight (bool): If True, adjust class weights inversely proportional to their frequencies.
    - smote_sampling_strategy (Union[None, float, str], optional): Strategy for applying SMOTE to address class imbalance.
    - top_n (int): Number of top features to select based on importance. Default is 60. 
    - skf (StratifiedKFold, optional): StratifiedKFold instance for splitting the dataset. If None, a default 
      StratifiedKFold configuration must be provided by the user.

    Returns:
    --------
    - pd.DataFrame: DataFrame limited to selected features based on the threshold and top N selection.

    Raises:
    -------
    - ValueError: If any of the columns specified in `exclude_cols` are not found in the DataFrame.
    - KeyError: If the `target_column` is not found in the DataFrame.
    - RuntimeError: If an error occurs during the training of the Logistic Regression model or during the calculation of SHAP values.

    Note:
    -----
    - This function assumes that the input DataFrame has already been cleaned and preprocessed.
    - The `skf` parameter, a StratifiedKFold instance, must be provided. If `skf` is not provided, the function will not work.

    Usage:
    ------
    >>> from sklearn.model_selection import StratifiedKFold
    >>> df = pd.read_csv("your_dataset.csv")
    >>> skf = StratifiedKFold(n_splits=5)
    >>> selected_features_df = feature_importances_logreg(
            df=df,
            target_column='target',
            exclude_cols=['id'],
            threshold_cut=0.01,
            use_class_weight=True,
            smote_sampling_strategy=0.5,
            top_n=50,
            skf=skf
        )
    >>> comparison_df
    """
    params = {
    'solver': 'liblinear',  # for small datasets and binary classification.
    'penalty': 'l2', 
    'C': 1.0,  # Inverse of regularization strength; smaller values specify stronger regularization.
    'max_iter': 100,  # Maximum number of iterations taken for the solvers to converge.
    'verbose': 0,  # For liblinear and lbfgs solvers, set verbose to any positive number for verbosity.
    }

    try:
        if skf is None:
            logging.error("StratifiedKFold (skf) is not provided. Please pass a valid StratifiedKFold instance.")
            return None
        
        # Check if target_column exists in the DataFrame
        if target_column not in df.columns:
            raise KeyError(f"The target column '{target_column}' was not found in the DataFrame.")

        # Check if any exclude column does not exist in DataFrame
        missing_cols = [col for col in exclude_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following columns to exclude are not in the DataFrame: {', '.join(missing_cols)}")

        df = df.reset_index(drop=True)
        X = df.drop(columns=[target_column] + exclude_cols)
        y = df[target_column]

        feature_importances = defaultdict(float)
        
        # Start Cross Validation
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx] 

            # Handle class imbalance with SMOTE if requested
            if smote_sampling_strategy and 0 < smote_sampling_strategy < 1:
                smote = SMOTE(random_state=42, sampling_strategy=smote_sampling_strategy)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            # Inside the cross-validation loop, before training the model
            params_copy = params.copy()

            # Dynamic use_class_weight adjustment
            if use_class_weight:
                params_copy['class_weight'] = 'balanced'
            
            # Train the model        
            model = LogisticRegression(**params_copy)
            model.fit(X_train, y_train)

            importances = np.abs(model.coef_[0])
            for i, col in enumerate(X_train.columns):
                feature_importances[col] += importances[i] / skf.n_splits

        # Normalize feature importances
        total_importance = sum(feature_importances.values())
        normalized_importances = {k: v / total_importance for k, v in feature_importances.items()}

        # Convert to DataFrame for easier manipulation
        feature_importances_df = pd.DataFrame(list(normalized_importances.items()), 
                                              columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
        
        # SHAP analysis
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_val, check_additivity=False)

        # Handling of SHAP values
        mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)

        mean_shap_df = pd.DataFrame({
            'Feature': X.columns, 
            'SHAP_Value': mean_abs_shap_values
        }).sort_values(by='SHAP_Value', ascending=False)
    
        comparison_df = mean_shap_df.merge(feature_importances_df, on='Feature', how='outer').fillna(0)
        if top_n:
            comparison_df = comparison_df.head(top_n)

        comparison_df = comparison_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        comparison_df['SHAP_Value'] = comparison_df['SHAP_Value'].round(4)
        comparison_df['Importance'] = comparison_df['Importance'].round(4)

        nonzero_features_df = comparison_df[comparison_df['Importance'] >= (threshold_cut if threshold_cut else 0)]
        nonzero_feature_names = nonzero_features_df['Feature'].tolist()
        
        df_selected_top_n = df[[*nonzero_feature_names, target_column] + exclude_cols]

        print('DataFrame before: ' + str(df.shape))
        print('DataFrame after: ' + str(df_selected_top_n.shape))
        print('\n')
        print(comparison_df)
        return df_selected_top_n

    except Exception as e:
        logging.error(f"An error occurred during the feature importance analysis: {e}")
        raise

# ====================== MODEL LGBM: Feature Importances & Selection =======================#

def feature_importances_lgbm(
    df: pd.DataFrame,
    target_column: str,
    exclude_cols: List[str],
    threshold_cut: Union[None, float, str] = None,
    use_class_weight: bool = False,
    smote_sampling_strategy: Union[None, float, str] = None,
    top_n: int = 60,
    skf: StratifiedKFold = None
) -> pd.DataFrame:
    """
    Feature Importance Evaluation Using LGBM Model and SHAP
    ==========================================================
    
    This function evaluates feature importance using LGBM Model and SHAP values, selecting top features 
    based on this analysis. It integrates handling class imbalance via SMOTE and allows for weighting classes 
    inversely proportional to their frequencies. The resulting DataFrame is limited to selected features 
    determined by importance scores and a specified threshold.

    Parameters:
    -----------
    - df (pd.DataFrame): Dataset for analysis.
    - target_column (str): Assuming 1 is the target class for binary classification.
    - exclude_cols (List[str]): Columns to exclude from analysis.
    - threshold_cut (Union[None, float, str], optional): Threshold for selecting features based on their importance. 
      Features with an importance score below this threshold will be excluded.
    - use_class_weight (bool): If True, adjust class weights inversely proportional to their frequencies.
    - smote_sampling_strategy (Union[None, float, str], optional): Strategy for applying SMOTE to address class imbalance.
    - top_n (int): Number of top features to select based on importance. Default is 60. 
    - skf (StratifiedKFold, optional): StratifiedKFold instance for splitting the dataset. If None, a default 
      StratifiedKFold configuration must be provided by the user.

    Returns:
    --------
    - pd.DataFrame: DataFrame limited to selected features based on the threshold and top N selection.

    Raises:
    -------
    - ValueError: If any of the columns specified in `exclude_cols` are not found in the DataFrame.
    - KeyError: If the `target_column` is not found in the DataFrame.
    - RuntimeError: If an error occurs during the training of the LGBM model or during the calculation of SHAP values.

    Note:
    -----
    - This function assumes that the input DataFrame has already been cleaned and preprocessed.
    - The `skf` parameter, a StratifiedKFold instance, must be provided. If `skf` is not provided, the function will not work.

    Usage:
    ------
    >>> from sklearn.model_selection import StratifiedKFold
    >>> df = pd.read_csv("your_dataset.csv")
    >>> skf = StratifiedKFold(n_splits=5)
    >>> selected_features_df = feature_importances_lgbm(
            df=df,
            target_column='target',
            exclude_cols=['id'],
            threshold_cut=0.01,
            use_class_weight=True,
            smote_sampling_strategy=0.5,
            top_n=50,
            skf=skf
        )
    >>> comparison_df
    """
    # Define a simple set of parameters for LightGBM
    params = {
        'verbosity': -1,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'aucpr',
        'is_unbalance':False
        #'class_weight': 'balanced' 
    }
    
    try:
        if skf is None:
            logging.error("StratifiedKFold (skf) is not provided. Please pass a valid StratifiedKFold instance.")
            return None
        
        # Check if target_column exists in the DataFrame
        if target_column not in df.columns:
            raise KeyError(f"The target column '{target_column}' was not found in the DataFrame.")

        # Check if any exclude column does not exist in DataFrame
        missing_cols = [col for col in exclude_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following columns to exclude are not in the DataFrame: {', '.join(missing_cols)}")

        df = df.reset_index(drop=True)
        X = df.drop(columns=[target_column] + exclude_cols)
        y = df[target_column]

        feature_importances = defaultdict(float)

        best_score = float('inf')
        best_model = None
        best_X_val, best_y_val = None, None
        
        # Start Cross Validation
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx] 

            # Handle class imbalance with SMOTE if requested
            if smote_sampling_strategy and 0 < smote_sampling_strategy < 1:
                smote = SMOTE(random_state=42, sampling_strategy=smote_sampling_strategy)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            # Inside the cross-validation loop, before training the model
            params_copy = params.copy()  # Create a copy of the initial parameter set

            # Dynamic scale_pos_weight adjustment
            if use_class_weight:
                num_negative = np.sum(y_train == 0)
                num_positive = np.sum(y_train == 1)
                scale_pos_weight_value = num_negative / num_positive
                params_copy['scale_pos_weight'] = scale_pos_weight_value 
            
            # Train the LightGBM model
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                # Prepare data for LGBM
                dtrain = lgb.Dataset(X_train, label=y_train)
                dval = lgb.Dataset(X_val, label=y_val)

                # Train the model
                evals_result = {}  # Initialize the dictionary outside the training function
                record_eval_callback = lgb.record_evaluation(evals_result)
                early_stopping_callback = lgb.early_stopping(stopping_rounds=100)
                
                model = lgb.train(
                    params=params_copy,
                    train_set=dtrain,
                    valid_sets=[dval],
                    valid_names=['validation'],
                    callbacks=[early_stopping_callback, record_eval_callback],
                    num_boost_round=params_copy.get('n_estimators', 100),
                    feval= lgbm_custom_eval('pr_auc'),
                )

            # Get feature importance using 'gain' and update dictionary
            importances = model.feature_importance(importance_type='gain')
            for i, col in enumerate(X_train.columns):
                feature_importances[col] += importances[i] / np.sum(importances)

            # Track the best model based on evaluation metric
            current_score = list(evals_result['validation'].values())[0][-1]
            if current_score < best_score:
                best_score = current_score
                best_model, best_X_val, best_y_val = model, X_val, y_val
        
        # Normalize feature importances
        total_importance = sum(feature_importances.values())
        normalized_importances = {k: v / total_importance for k, v in feature_importances.items()}

        # Convert to DataFrame 
        feature_importances_df = pd.DataFrame(list(normalized_importances.items()), 
                                              columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
        
        # SHAP analysis
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_val, check_additivity=False)

        # Handling of SHAP values
        mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)

        mean_shap_df = pd.DataFrame({
            'Feature': X.columns, 
            'SHAP_Value': mean_abs_shap_values
        }).sort_values(by='SHAP_Value', ascending=False)
    
        comparison_df = mean_shap_df.merge(feature_importances_df, on='Feature', how='outer').fillna(0)
        if top_n:
            comparison_df = comparison_df.head(top_n)
        
        comparison_df = comparison_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        comparison_df['SHAP_Value'] = comparison_df['SHAP_Value'].round(4)
        comparison_df['Importance'] = comparison_df['Importance'].round(4)

        nonzero_features_df = comparison_df[comparison_df['Importance'] >= (threshold_cut if threshold_cut else 0)]
        nonzero_feature_names = nonzero_features_df['Feature'].tolist()
        
        df_selected_top_n = df[[*nonzero_feature_names, target_column] + exclude_cols]

        print('DataFrame before: ' + str(df.shape))
        print('DataFrame after: ' + str(df_selected_top_n.shape))
        print('\n')
        print(comparison_df)
        return df_selected_top_n

    except Exception as e:
        logging.error(f"An error occurred during the feature importance analysis: {e}")
        raise

# ====================== MODEL XGBoost: Feature Importances & Selection =======================#

def feature_importances_xgb(
    df: pd.DataFrame,
    target_column: str,
    exclude_cols: List[str],
    threshold_cut: Union[None, float, str] = None,
    use_class_weight: bool = False,
    smote_sampling_strategy: Union[None, float, str] = None,
    top_n: int = 60,
    skf: StratifiedKFold = None
) -> pd.DataFrame:
    """
    Feature Importance Evaluation Using XGBoost Model and SHAP
    ==========================================================
    
    This function evaluates feature importance using XGBoost Model and SHAP values, selecting top features 
    based on this analysis. It integrates handling class imbalance via SMOTE and allows for weighting classes 
    inversely proportional to their frequencies. The resulting DataFrame is limited to selected features 
    determined by importance scores and a specified threshold.

    Parameters:
    -----------
    - df (pd.DataFrame): Dataset for analysis.
    - target_column (str): Assuming 1 is the target class for binary classification.
    - exclude_cols (List[str]): Columns to exclude from analysis.
    - threshold_cut (Union[None, float, str], optional): Threshold for selecting features based on their importance. 
      Features with an importance score below this threshold will be excluded.
    - use_class_weight (bool): If True, adjust class weights inversely proportional to their frequencies.
    - smote_sampling_strategy (Union[None, float, str], optional): Strategy for applying SMOTE to address class imbalance.
    - top_n (int): Number of top features to select based on importance. Default is 60. 
    - skf (StratifiedKFold, optional): StratifiedKFold instance for splitting the dataset. If None, a default 
      StratifiedKFold configuration must be provided by the user.

    Returns:
    --------
    - pd.DataFrame: DataFrame limited to selected features based on the threshold and top N selection.

    Raises:
    -------
    - ValueError: If any of the columns specified in `exclude_cols` are not found in the DataFrame.
    - KeyError: If the `target_column` is not found in the DataFrame.
    - RuntimeError: If an error occurs during the training of the XGBoost model or during the calculation of SHAP values.

    Note:
    -----
    - This function assumes that the input DataFrame has already been cleaned and preprocessed.
    - The `skf` parameter, a StratifiedKFold instance, must be provided. If `skf` is not provided, the function will not work.

    Usage:
    ------
    >>> from sklearn.model_selection import StratifiedKFold
    >>> df = pd.read_csv("your_dataset.csv")
    >>> skf = StratifiedKFold(n_splits=5)
    >>> selected_features_df = feature_importances_xgb(
            df=df,
            target_column='target',
            exclude_cols=['id'],
            threshold_cut=0.01,
            use_class_weight=True,
            smote_sampling_strategy=0.5,
            top_n=50,
            skf=skf
        )
    >>> comparison_df
    """
    # Set XGBoost parameters
    params = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
    }
    
    try:
        if skf is None:
            logging.error("StratifiedKFold (skf) is not provided. Please pass a valid StratifiedKFold instance.")
            return None
        
        # Check if target_column exists in the DataFrame
        if target_column not in df.columns:
            raise KeyError(f"The target column '{target_column}' was not found in the DataFrame.")

        # Check if any exclude column does not exist in DataFrame
        missing_cols = [col for col in exclude_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following columns to exclude are not in the DataFrame: {', '.join(missing_cols)}")

        df = df.reset_index(drop=True)
        X = df.drop(columns=[target_column] + exclude_cols)
        y = df[target_column]

        feature_importances = defaultdict(float)

        best_score = float('inf')
        best_model = None
        best_X_val, best_y_val = None, None
        
        # Start Stratified K-Fold Cross-Validation
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]  

            # Handle class imbalance with SMOTE if requested
            if smote_sampling_strategy and 0 < smote_sampling_strategy < 1:
                smote = SMOTE(random_state=42, sampling_strategy=smote_sampling_strategy)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            
            # Dynamic scale_pos_weight adjustment
            if use_class_weight:
                num_negative = np.sum(y_train == 0)
                num_positive = np.sum(y_train == 1)
                scale_pos_weight_value = num_negative / num_positive
                params['scale_pos_weight'] = scale_pos_weight_value * 0.5
                
            # Prepare data for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())
            
            # Train the model
            evals_result = {}
            model = xgb.train(
                params,
                dtrain,
                evals=[(dval, 'eval')],
                early_stopping_rounds=100,
                num_boost_round=params.get('n_estimators', 100),
                evals_result=evals_result,
                verbose_eval=False
            )
            
            # Update feature importances
            importances = model.get_score(importance_type='gain')
            for feature, importance in importances.items():
                feature_importances[feature] += importance / skf.n_splits
            
            # Track the best model based on evaluation metric
            current_score = list(evals_result['eval'].values())[0][-1]
            if current_score < best_score:
                best_score = current_score
                best_model, best_X_val, best_y_val = model, X_val, y_val
        
        # Normalize feature importances
        total_importance = sum(feature_importances.values())
        normalized_importances = {k: v / total_importance for k, v in feature_importances.items()}
        
        # Convert to DataFrame 
        feature_importances_df = pd.DataFrame(list(normalized_importances.items()), 
                                              columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
        
        # SHAP analysis
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_val)

        # Handling of SHAP values
        mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)

        mean_shap_df = pd.DataFrame({
            'Feature': X.columns, 
            'SHAP_Value': mean_abs_shap_values
        }).sort_values(by='SHAP_Value', ascending=False)
    
        comparison_df = mean_shap_df.merge(feature_importances_df, on='Feature', how='outer').fillna(0)
        if top_n:
            comparison_df = comparison_df.head(top_n)
        
        comparison_df = comparison_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        comparison_df['SHAP_Value'] = comparison_df['SHAP_Value'].round(4)
        comparison_df['Importance'] = comparison_df['Importance'].round(4)

        nonzero_features_df = comparison_df[comparison_df['Importance'] >= (threshold_cut if threshold_cut else 0)]
        nonzero_feature_names = nonzero_features_df['Feature'].tolist()
        
        df_selected_top_n = df[[*nonzero_feature_names, target_column] + exclude_cols]

        print('DataFrame before: ' + str(df.shape))
        print('DataFrame after: ' + str(df_selected_top_n.shape))
        print('\n')
        print(comparison_df)
        return df_selected_top_n

    except Exception as e:
        logging.error(f"An error occurred during the feature importance analysis: {e}")
        raise

# ====================== MODEL LOGREG: Hyperparameter Tuning Functions =======================#
# ---------------------- MODEL LOGREG: Optuna optimization function ---------------------- #

def logreg_objective(trial, X, y, 
                     params_space, skf, 
                     metric=None, 
                     use_class_weight=False, 
                     smote_sampling_strategy=None):
    """
    Objective function to optimize Logistic Regression parameters using Optuna.

    Parameters:
    - trial: Optuna trial object for hyperparameter tuning.
    - X, y: Features and target datasets.
    - params_space: Dictionary of search spaces for parameters 
    - skf: StratifiedKFold object for splitting the dataset.
    - metric: Metric to optimize.
    - use_class_weight: Whether to adjust class weights.
    - smote_sampling_strategy: SMOTE sampling strategy.

    Returns:
    - Average score of the optimization metric across all folds.
    """
    try: 
        params = {}

        # Dynamically add parameters from params_space
        param_keys = ['solver', 'penalty', 'C', 'max_iter']
        
        # Log if there are extra parameters not expected
        extra_params = set(params_space.keys()) - set(param_keys)
        if extra_params:
            logging.warning(f"Extra parameters that aren't used in the model: {extra_params}. Please use 'solver', 'penalty', 'C', 'max_iter'.")
   
        # Check and add only the parameters that are expected and provided
        for key in param_keys:
            if key in params_space:
                if key in ['solver', 'penalty']:
                    params[key] = trial.suggest_categorical(key, params_space[key])
                elif key in ['C']:
                    params[key] = trial.suggest_float(key, *params_space[key])
                else:
                    params[key] = trial.suggest_int(key, *params_space[key])
            else:
                logging.info(f"Parameter '{key}' not provided in params_space and will be skipped.")

        # Initialize the list to store scores for each fold
        scores = []

        # Start Cross Validation
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Handle class imbalance with SMOTE if requested
            if smote_sampling_strategy and 0 < smote_sampling_strategy < 1:
                smote = SMOTE(random_state=42, sampling_strategy=smote_sampling_strategy)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            
            # Make a copy of param_grid to modify
            params_copy = params.copy()   

            # Dynamic use_class_weight adjustment
            if use_class_weight:
                # Update 'class_weight' in the hyperparameters
                params_copy['class_weight'] = trial.suggest_categorical('class_weight', [None, 'balanced'])

            # Fit logistic regression for feature selection
            model = LogisticRegression(**params_copy)
            model.fit(X_train, y_train)

            # Generate predictions and probabilities using best iteration
            y_pred_prob = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_prob > 0.5).astype(int) 

            # Evaluate the model
            score = calculate_metric(y_val, y_pred, y_pred_prob, metric)
            scores.append(score)

        # End Cross Validation # Log and return the average score across all folds
        average_score = np.mean(scores)
        #logging.info(f"Trial completed with score: {average_score}")

        return average_score
    except Exception as e:
        logging.error(f"Error in trial: {e}")
        raise

# ======================= MODEL LOGREG: Hyperparameter tuning using Optuna ======================= #

def hyperparameter_tuning_logreg(df,  target_column=None, exclude_cols=None,
                                params=None, skf=None, 
                                n_trials=None, metric=None, 
                                threshold=0.96, 
                                n_trials_no_improve=50, 
                                use_class_weight=False, 
                                smote_sampling_strategy=None):
    """
    Hyperparameter Tuning for Logistic Regression Using Optuna
    ==========================================================

    This function configures and executes a hyperparameter search using Optuna to find the optimal 
    settings for a logistic regression model based on a specified performance metric. It supports 
    handling class imbalances with optional SMOTE and class weighting, and it uses StratifiedKFold 
    for cross-validation to ensure the dataset's distribution is respected.

    Sections:
    ---------
    - Overview
    - Parameters
    - Returns
    - Raises
    - Usage
    - Notes

    Overview
    --------
    The function performs hyperparameter tuning for a Logistic Regression model by utilizing the Optuna
    optimization framework to systematically explore a range of parameter configurations. It integrates
    several advanced techniques such as class weight adjustment and SMOTE for managing class imbalance,
    and utilizes cross-validation to ensure robust evaluation of model performance.

    Parameters
    ----------
    - df (pd.DataFrame): The complete dataset containing both features and the target variable.
    - target_column (str, optional): Name of the target variable column in the dataframe.
    - exclude_cols (list, optional): List of column names to exclude from the feature set.
    - params (dict): A dictionary defining the search space for logistic regression parameters.
                     Each entry should define a range or list of options that Optuna will explore.
    - skf (StratifiedKFold, optional): A StratifiedKFold instance for cross-validation. If None,
                                       the function requires an externally provided StratifiedKFold instance.
    - n_trials (int, optional): The number of trials that Optuna will perform. Each trial tests a 
                                different combination of parameters.
    - metric (str, optional): The performance metric to optimize. Valid options include 'f1', 'balanced_acc',
                              'roc_auc', 'pr_auc', 'recall'. The choice of metric influences how the 
                              model's performance is evaluated.
    - threshold (float, optional): A performance threshold for early stopping. If a trial achieves 
                                   a performance above this threshold, the search can optionally be 
                                   halted early.
    - n_trials_no_improve (int, optional): The number of consecutive trials without improvement in 
                                           performance to tolerate before stopping early. Helps in 
                                           limiting resource waste.
    - use_class_weight (bool, optional): Flag to determine whether to adjust class weights inversely 
                                         proportional to class frequencies.
    - smote_sampling_strategy (float, optional): Specifies the desired ratio of the number of samples 
                                                 in the minority class over the majority class after 
                                                 resampling with SMOTE. Must be between 0 and 1.

    Returns
    -------
    - best_params (dict): Dictionary of the best parameters found during the optimization.

    Raises
    ------
    - ValueError: If 'df' is not a pandas DataFrame, or other parameters are not in the expected format.
    - RuntimeError: If the function encounters an issue during the optimization process.

    Usage
    -----
    >>> best_params = hyperparameter_tuning_logreg(df=data, target_column='outcome', params=params_dict,
                                                   skf=StratifiedKFold(n_splits=5), n_trials=100,
                                                   metric='roc_auc', use_class_weight=True)

    Notes
    -----
    - The function relies heavily on the configuration provided via 'params' and 'skf'. Incorrect
      configuration may lead to suboptimal tuning results or runtime errors.
    - It is assumed that the data passed to the function does not contain any missing values in 
      the features or target columns.
    """
    if skf is None or params is None:
        logging.error("StratifiedKFold (skf) or parameters (params) not provided. Please pass valid instances.")
        return None
    
    # Call the function to validate input DataFrame and prepare the data for model training
    X, _ = cu.validate_and_prepare(df, target_column, exclude_cols)
    y = df[target_column] 

    logging.info("Starting hyperparameter tuning for Logistic Regression model...")

    trials_without_improvement = 0
    best_score = float('-inf')

    def early_stopping_callback(study, trial):
        nonlocal best_score, trials_without_improvement
        if trial.value > best_score:
            best_score = trial.value
            trials_without_improvement = 0
        else:
            trials_without_improvement += 1

        if threshold is not None and best_score >= threshold:
            study.stop()

        if n_trials_no_improve is not None and trials_without_improvement >= n_trials_no_improve:
            study.stop()
    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: logreg_objective(trial, X, y, params, skf, metric, use_class_weight, smote_sampling_strategy), 
                                                    n_trials = n_trials, callbacks=[early_stopping_callback], 
                                                    n_jobs= -1
                                                    )

        best_params = study.best_params

        print("\nBest Parameters for Logistic Regression:")
        for key, value in best_params.items():
            print(f"{key}: {value}")

        logging.info("Hyperparameter tuning completed. Best parameters: %s", best_params)
        return best_params

    except Exception as e:
        logging.error(f"Error in hyperparameter tuning: {e}")
        raise

# ======================= MODEL LOGREG: Training & Cross Validation ======================= #

# ---------------------- MODEL LOGREG: Train the model ---------------------- #
def train_model_logreg(X_train, y_train, X_val, y_val, hyperparams):
    """
    Trains a Logistic Regression model with the given hyperparameters.

    Parameters:
    - X_train (DataFrame): Training features.
    - y_train (Series): Training target variable.
    - X_val (DataFrame): Validation features.
    - y_val (Series): Validation target variable.
    - hyperparams (dict): Hyperparameters for the Logistic Regression model.

    Returns:
    - model: The trained Logistic Regression model.
    """
    # Define default hyperparameters
    default_hyperparams = {
        'verbose': 0
    }
    # Validate and merge hyperparameters
    if hyperparams is not None:
        if not isinstance(hyperparams, dict):
            raise TypeError("Hyperparameters must be a dictionary.")
        # Merge user-specified hyperparams with defaults, user-specified taking precedence
        hyperparams = {**default_hyperparams, **hyperparams}
    else:
        hyperparams = default_hyperparams.copy()
    
    try:
        # Print hyperparameters for debugging
        logging.info(f"Hyperparameters being used:{hyperparams}")

        model = LogisticRegression(**hyperparams)
        model.fit(X_train, y_train)

        return model
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        return None
    
# ---------------------- MODEL LOGREG: Performs training and evaluation for one fold in cross-validation ---------------------- #

def perform_fold_logreg(X_train, y_train, X_val, y_val, 
                        hyperparams, fold_number,
                        smote_sampling_strategy=None):
    """
    Performs training and evaluation for one fold in cross-validation directly using data frames.
    
    This function handles the training and evaluation for a single fold in the cross-validation process.
    It trains the model using the provided training data and hyperparameters and evaluates it on both
    the training and validation sets. It returns the performance metrics for both training and validation.
    
    Parameters:
    - X_train (DataFrame): Training features DataFrame.
    - y_train (Series): Training target variable Series.
    - X_val (DataFrame): Validation features DataFrame.
    - y_val (Series): Validation target variable Series.
    - hyperparams (dict): Hyperparameters for the Logistic Regression model.
    - fold_number (int): The fold number being processed (for logging and tracking purposes).
    - smote_sampling_strategy (float): Fraction for SMOTE sampling strategy (between 0 and 1).
    
    Returns:
    - tuple: A tuple containing two dictionaries (train_metrics, val_metrics), where each dictionary
             includes various performance metrics such as precision, recall, F1 score, AUC-ROC, etc.
    
    Raises:
    - RuntimeError: If the model training fails.
    - Exception: Captures and logs any other exceptions that occur during fold execution.
    """
    try:
        logging.info(f"Starting with fold {fold_number}:")
        logging.info(f"Class distribution in training set (Fold {fold_number}): {y_train.value_counts()}")
    
        # Handle class imbalance with SMOTE if requested
        if smote_sampling_strategy and 0 < smote_sampling_strategy < 1:
            smote = SMOTE(random_state=42, sampling_strategy=smote_sampling_strategy)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logging.info(f"SMOTE applied with strategy {smote_sampling_strategy} on fold {fold_number}.")

        # Model Training 
        model = train_model_logreg(X_train, y_train, X_val, y_val, hyperparams)
        if model is None:
            logging.error("Model training failed for fold {fold_number}")
            return None, None

        # Generate probabilities for evaluation
        y_train_pred_prob = model.predict_proba(X_train)[:, 1]
        y_val_pred_prob = model.predict_proba(X_val)[:, 1]

        y_train_pred = (y_train_pred_prob >= 0.5).astype(int)
        y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

        # After training and evaluation, log the validation predicted probabilities
        logging.info(f"Validation predicted probabilities for fold {fold_number}: {y_val_pred_prob[:5]}")
        logging.info(f"Fold {fold_number} training completed.\n")


        return (y_train, y_train_pred, y_train_pred_prob), (y_val, y_val_pred, y_val_pred_prob)
    
    except Exception as e:
        # Print any exceptions that occur during fold execution
        logging.error(f"Error in fold execution: {e}")
        # Return None to indicate failure in processing this fold
        return None, None, None, None

# ---------------------- MODEL LOGREG: Main Training and Evaluation Function ---------------------- #

def train_evaluate_logreg_cv(df, skf=None, target_column=None, 
                             hyperparams=None, exclude_cols=None,
                             smote_sampling_strategy=None):
    """
    Training and Evaluation for Logistic Regression using Cross-Validation
    ======================================================================

    This function orchestrates the training and evaluation of a Logistic Regression model using cross-validation. 
    It sets up the cross-validation, prepares the data, and calls `perform_fold_logreg` for each fold. 
    It collects the results from all folds, computes the average metrics across all folds, and returns 
    these aggregated metrics.

    Sections:
    ---------
    - Overview
    - Parameters
    - Returns
    - Raises
    - Usage
    - Notes

    Overview
    --------
    The function performs the entire cross-validation process for a Logistic Regression model. 
    It ensures that the data is properly split into training and validation sets, trains the model on each fold, 
    and evaluates its performance. The results from all folds are aggregated to provide an overall 
    performance evaluation of the model.

    Parameters
    ----------
    - df (pd.DataFrame): The complete dataset containing both features and the target variable.
    - skf (StratifiedKFold): A StratifiedKFold instance for cross-validation.
    - target_column (str): Name of the target variable column in the dataframe.
    - hyperparams (dict, optional): Hyperparameters for the Logistic Regression model.
    - exclude_cols (list, optional): List of columns to exclude from features.
    - smote_sampling_strategy (float, optional): Specifies the desired ratio of the number of samples 
                                                 in the minority class over the majority class after 
                                                 resampling with SMOTE. Must be between 0 and 1.

    Returns
    -------
    - avg_metrics_across_folds (dict): Averaged evaluation metrics across all folds.
    - std_metrics_across_folds (dict): Standard deviation of evaluation metrics across all folds.
    - all_y_val (list): Concatenated list of all validation target values from each fold.
    - all_y_pred_prob (list): Concatenated list of all model prediction probabilities for the validation data from each fold.

    Raises
    ------
    - ValueError: If 'df' is not a pandas DataFrame, or other parameters are not in the expected format.
    - RuntimeError: If the function encounters an issue during the cross-validation process.

    Usage
    -----
    >>> avg_metrics, std_metrics, all_y_val, all_y_pred_prob, train_metrics_df, val_metrics_df = train_evaluate_logreg_cv(
            df=data, skf=StratifiedKFold(n_splits=5), target_column='outcome', 
            hyperparams=best_params_logreg, exclude_cols=['ID'], smote_sampling_strategy=0.5)

    Notes
    -----
    - The function relies on the configuration provided via 'hyperparams' and 'skf'. Incorrect 
      configuration may lead to suboptimal training results or runtime errors.
    - It is assumed that the data passed to the function does not contain any missing values in 
      the features or target columns.
    - SMOTE is applied if the 'smote_sampling_strategy' parameter is provided and valid.
    """
    if skf is None or hyperparams is None:
        logging.error("StratifiedKFold (skf) or hyperparameters (dict) not provided. Please pass valid instances.")
        return None

    # Check if hyperparams are provided and valid
    logging.info(f"Received hyperparams: {hyperparams}, Type: {type(hyperparams)}\n")

    # Call the function to validate input DataFrame and prepare the data for model training
    X, _ = cu.validate_and_prepare(df, target_column, exclude_cols)
    y = df[target_column]  # Assumes y is already numeric and suitable for stratification

    # Initialize
    all_train_results = []
    all_val_results = []
    all_y_val = []
    all_y_pred_prob = []

    # Start Cross Validation
    for fold_number, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Ensure the fold is processed correctly (implement within perform_fold_logreg)
        fold_train_result, fold_val_result = perform_fold_logreg(
            X_train, y_train, 
            X_val, y_val_fold, 
            hyperparams.copy(), 
            fold_number, 
            smote_sampling_strategy)

        if fold_train_result and fold_val_result:
            all_train_results.append(fold_train_result)
            all_val_results.append(fold_val_result)
            all_y_val.extend(y_val_fold)  # Aggregating true validation labels
            all_y_pred_prob.extend(fold_val_result[2])  # Aggregating validation prediction probabilities

    # After the cross-validation loop, log the aggregated validation predicted probabilities
    logging.info(f"AGGREGATED validation predicted probabilities: {all_y_pred_prob[:5]}")
    logging.info("Cross Validation completed.")

    # Evaluate all folds together
    avg_metrics_across_folds, std_metrics_across_folds, train_metrics_df, val_metrics_df = evaluate_model(all_train_results, all_val_results)

    return avg_metrics_across_folds, std_metrics_across_folds, all_y_val, all_y_pred_prob, train_metrics_df, val_metrics_df 



# ---------------------- HELPER FUNCTION: Calculate metrics ---------------------- #
def calculate_metric(y_true, y_pred, y_pred_proba, metric):
    """
    Calculate the specified metric.
    
    Parameters:
    - y_true: Actual target values.
    - y_pred: Predicted target values.
    - y_pred_prob: Predicted probabilities.
    - metric: The metric to calculate ('f1', 'balanced_acc', 'roc_auc', 'pr_auc', 'recall').
    
    Returns:
    - Calculated metric.
    """

    if metric == 'f1':
        return f1_score(y_true, y_pred)
    elif metric == 'balanced_acc':
        return balanced_accuracy_score(y_true, y_pred)
    elif metric == 'roc_auc':
        return roc_auc_score(y_true, y_pred_proba)
    elif metric == 'pr_auc':
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        return auc(recall, precision)
    elif metric == 'recall':
        return recall_score(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported optimization metric: {metric}")

# ----------------- Calculate and display metrics for training & validation set -------------- #

def calculate_and_display_metrics(y_train, y_train_pred, y_train_pred_prob, 
                                  y_val, y_val_pred, y_val_pred_prob, model_name='Model'):
    """
    Calculates and displays evaluation metrics for the training and validation sets based on provided predictions.
    
    Parameters:
    - y_train (pd.Series): Training target variable.
    - y_train_pred (np.array): Training predictions (binary).
    - y_train_pred_prob (np.array): Training prediction probabilities.
    - y_val (pd.Series): Validation target variable.
    - y_val_pred (np.array): Validation predictions (binary).
    - y_val_pred_prob (np.array): Validation prediction probabilities.
    - model_name (str, optional): Name of the model for display purposes.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the calculated metrics for both training and validation sets.
    """
    try:
        # Calculate metrics for the training set
        precision_train = precision_score(y_train, y_train_pred, zero_division=0)
        recall_train = recall_score(y_train, y_train_pred)
        f1_train = f1_score(y_train, y_train_pred)
        auc_roc_train = roc_auc_score(y_train, y_train_pred_prob)
        balanced_acc_train = balanced_accuracy_score(y_train, y_train_pred)
        pr_curve_train = precision_recall_curve(y_train, y_train_pred_prob)
        pr_auc_train = auc(pr_curve_train[1], pr_curve_train[0])

        # Calculate metrics for the validation set
        precision_val = precision_score(y_val, y_val_pred, zero_division=0)
        recall_val = recall_score(y_val, y_val_pred)
        f1_val = f1_score(y_val, y_val_pred)
        auc_roc_val = roc_auc_score(y_val, y_val_pred_prob)
        balanced_acc_val = balanced_accuracy_score(y_val, y_val_pred)
        pr_curve_val = precision_recall_curve(y_val, y_val_pred_prob)
        pr_auc_val = auc(pr_curve_val[1], pr_curve_val[0])

        # Compile metrics into a DataFrame for display
        metrics_df = pd.DataFrame({
            'Metric': ['precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'balanced_acc'],
            'Train Set': [precision_train, 
                            recall_train,
                            f1_train, 
                            pr_auc_train, 
                            auc_roc_train, 
                            balanced_acc_train],
            'Test Set': [precision_val, 
                               recall_val, 
                               f1_val, 
                               pr_auc_val, 
                               auc_roc_val, 
                               balanced_acc_val]
        })

        # Display the metrics DataFrame
        print(f"\n{model_name} - Comparison of Train & Test Metrics:\n")
        print(metrics_df.to_string(index=False))

        return metrics_df
    except Exception as e:
        logging.exception("Failed to calculate or display metrics.")
        raise ValueError("An error occurred during the metrics calculation and display.") from e

    
# ---------------------- HELPER FUNCTION: Evaluation of model using several metrics ---------------------- #
def versatile_custom_eval(y_true, y_pred_proba, metric):
    """
    A versatile custom evaluation function that supports multiple metrics.

    Parameters:
    - y_true (array-like): The ground truth labels.
    - y_pred_proba (array-like): The predicted probabilities from the model.
    - metric (str): The metric to calculate ('f1', 'roc_auc', 'balanced_acc', 'recall', 'pr_auc').

    Returns:
    - tuple: A tuple containing the metric name, the calculated metric value, and a boolean indicating
             whether a higher value is better.

    This function is designed to be used with a model as a custom evaluation function for various metrics.
    It supports extensive error handling and logging for robust and flexible usage.
    """
    try:
        # Threshold probabilities to get binary predictions
        y_pred = np.where(y_pred_proba >= 0.5, 1, 0)

        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'roc_auc':
            score = roc_auc_score(y_true, y_pred_proba)
        elif metric == 'balanced_acc':
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        elif metric == 'pr_auc':
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            score = auc(recall, precision)
        else:
            logging.error(f'Unsupported metric: {metric}')
            raise ValueError(f'Unsupported metric: {metric}')

        # Log the successful calculation
        #logging.info(f'Successfully calculated {metric} score: {score}')
        
        # Note: is_higher_better is a boolean indicating if a higher score is better
        is_higher_better = True if metric not in ['log_loss'] else False
        return (metric, score, is_higher_better)

    except Exception as e:
        logging.error(f'Error calculating {metric}: {e}')
        raise

# ---------------------- MODEL LGBM: compatible custom evaluation function ---------------------- #
def lgbm_custom_eval(metric):
    """
    For binary classification. Wrapper function to create a LightGBM compatible custom evaluation function for a given metric.

    Parameters:
    - metric (str): The metric to calculate ('f1', 'roc_auc', 'balanced_acc', 'recall', 'pr_auc').

    Returns:
    - function: A function that can be passed to LightGBM's `feval` parameter during model training.
    """
    def eval_function(y_pred, dtrain):
        # Obtain the true labels from the LightGBM dataset
        y_true = dtrain.get_label()

        # Ensure y_pred is correctly interpreted as probabilities
        if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            y_pred_proba = y_pred[:, 1]  # Use probabilities for the positive class
        else:
            y_pred_proba = y_pred  # Assume y_pred is already the probability of the positive class

        # Call the versatile custom eval function with probabilities and actual labels
        metric_name, score, is_higher_better = versatile_custom_eval(y_true, y_pred_proba, metric)
        
        return metric_name, score, is_higher_better
    return eval_function

# ---------------------- MODEL Logistic Regression: compatible custom evaluation function ---------------------- #
def logreg_custom_eval(metric):
    """
    For binary classification. Function to apply a custom evaluation metric to logistic regression model predictions.

    Parameters:
    - metric (str): The metric to calculate ('f1', 'roc_auc', 'balanced_acc', 'recall', 'pr_auc').

    Returns:
    - function: A function that accepts true labels and predicted probabilities, returning the metric name, score, and whether higher is better.
    """
    def eval_function(y_true, y_pred_proba):
        # Ensure y_pred_proba is correctly interpreted as probabilities
        # This assumes y_pred_proba is already the probability of the positive class
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]

        # Call the versatile custom eval function with probabilities and actual labels
        metric_name, score, is_higher_better = versatile_custom_eval(y_true, y_pred_proba, metric)
        
        return metric_name, score, is_higher_better
    return eval_function


# ---------------------- MODEL XGBoost: compatible custom evaluation function ---------------------- #
def xgb_custom_eval(metric):
    """
     For binary classification. Wrapper function to create an XGBoost compatible custom evaluation function for a given metric.

    Parameters:
    - metric (str): The metric to calculate ('f1', 'roc_auc', 'balanced_acc', 'recall', 'pr_auc').

    Returns:
    - function: A function that can be passed to XGBoost's `feval` parameter during model training.
    """
    def eval_function(y_pred, dtrain):
        # Obtain the true labels
        y_true = dtrain.get_label()

        # Ensure y_pred is correctly interpreted as probabilities for binary classification
        # XGBoost outputs raw scores by default, so we apply a sigmoid function for logistic regression tasks
        if y_pred.ndim == 1:
            # Convert raw scores to probabilities if necessary
            y_pred_proba = 1 / (1 + np.exp(-y_pred))
        else:
            y_pred_proba = y_pred

        # Calculate the specified metric
        metric_name, score, is_higher_better = versatile_custom_eval(y_true, y_pred_proba, metric)
        
        # XGBoost expects a tuple of (str, float) for the custom evaluation metric
        return metric_name, score
    return eval_function

# ---------------------- MODEL: Evaluates the given model using several metrics ---------------------- #

def evaluate_model(train_results, val_results):
    """
    This function calculates and displays several evaluation metrics for the training and validation sets.

    Parameters:
    - train_results (list of tuples): A list where each tuple contains (y_true, y_pred, y_pred_prob) for the training set.
    - val_results (list of tuples): A list where each tuple contains (y_true, y_pred, y_pred_prob) for the validation set.

    Returns:
    - avg_metrics_across_folds (pd.DataFrame): A DataFrame containing the average of the evaluation metrics across all folds.
    - std_metrics_across_folds (pd.DataFrame): A DataFrame containing the standard deviation of the evaluation metrics across all folds.
    - train_metrics_df (pd.DataFrame): Detailed DataFrame of metrics for each training fold.
    - val_metrics_df (pd.DataFrame): Detailed DataFrame of metrics for each validation fold.
    """
    try:
        # Process results and compute metrics for each fold
        def compute_metrics(results):
            metrics_list = []
            for y_true, y_pred, y_pred_prob in results:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
                pr_auc = auc(recall, precision)

                metrics = {
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred),
                    'f1': f1_score(y_true, y_pred),
                    'pr_auc': pr_auc,
                    'roc_auc': roc_auc_score(y_true, y_pred_prob),
                    'balanced_acc': balanced_accuracy_score(y_true, y_pred)
                }
                metrics_list.append(metrics)
            return metrics_list

        # Check if train_results and val_results are not empty
        if not train_results or not val_results:
            raise ValueError("train_results and val_results cannot be empty.")
        
        # Compute metrics for training and validation results
        train_metrics_list = compute_metrics(train_results)
        val_metrics_list = compute_metrics(val_results)

        # Convert to DataFrame for easy manipulation
        train_metrics_df = pd.DataFrame(train_metrics_list)
        val_metrics_df = pd.DataFrame(val_metrics_list)

        # Calculate average and standard deviation of metrics
        avg_metrics = pd.concat([train_metrics_df.mean(), val_metrics_df.mean()], axis=1)
        std_metrics = pd.concat([train_metrics_df.std(), val_metrics_df.std()], axis=1)
        avg_metrics.columns = ['Train Avg', 'Validation Avg']
        std_metrics.columns = ['Train Std', 'Validation Std']

        # Displaying metrics
        print("\nTraining Metrics for Each Fold:")
        print(train_metrics_df)
        print("\nValidation Metrics for Each Fold:")
        print(val_metrics_df)

        print("\nAverage Metrics across folds:\n", avg_metrics)
        print("\nStandard Deviation of Metrics:\n", std_metrics)

        return avg_metrics, std_metrics, train_metrics_df, val_metrics_df

    except Exception as e:
        # Log the error and return placeholder data
        logging.error(f"Error in evaluate_model: {e}")
        empty_avg_std_df = pd.DataFrame(columns=['Train Avg', 'Validation Avg', 'Train Std', 'Validation Std'])
        empty_metrics_df = pd.DataFrame()
        return empty_avg_std_df, empty_avg_std_df, empty_metrics_df, empty_metrics_df

# ---------------------- MODEL TRAINING: PLOTS EVALUATION METRICS ---------------------- #

def plot_evals_results(evals_results, trial_number, metric, model_name="Model",
                            figsize=(7, 5), show_fig=True, patience=None):
    """
    Plots the training and validation evaluation metrics over training rounds for a given trial,
    with a vertical line indicating the early stopping point based on the specified metric.
    Early stopping occurs when the specified metric does not improve for a given number of consecutive rounds (patience).

    Parameters:
    - evals_results (dict): Evaluation results from the model training process for each trial.
    - trial_number (int): Trial number to plot results for.
    - metric (str): Metric to plot, e.g., 'logloss', 'auc', 'aucpr'.
    - model_name (str): Name of the model to include in the plot title.
    - figsize (tuple): Size of the figure to be plotted.
    - show_fig (bool): If True, the figure will be shown.
    - patience (int): Number of rounds to wait for improvement in the metric before stopping.

    Raises:
    - ValueError: If the provided `evals_results` for the trial does not contain the expected structure or metrics.
    - KeyError: If the specified metric is not available in the `evals_results` for the trial.
    """
    try:
        if trial_number not in evals_results:
            raise ValueError(f"No evaluation results found for trial number {trial_number}.")
 
        trial_evals = evals_results[trial_number]
        eval_set_name = 'validation' if 'validation' in trial_evals else 'test'
        if 'train' not in trial_evals or eval_set_name not in trial_evals:
            raise ValueError(f"Evaluation results for trial number {trial_number} must contain both 'train' and '{eval_set_name}' keys.")
        
        train_metric = trial_evals['train'][metric]
        val_metric = trial_evals[eval_set_name][metric]

        # Find the round where the improvement stops based on the patience level
        early_stopping_round = None
        if metric in ['auc', 'aucpr', 'f1']:  # Higher is better
            for i in range(len(val_metric)-patience):
                # If there is no increase in the next 'patience' rounds, stop
                if not any(val_metric[i] < val_metric[i+j] for j in range(1, patience+1)):
                    early_stopping_round = i
                    break
        else:  # Lower is better, e.g., 'logloss'
            for i in range(len(val_metric)-patience):
                # If there is no decrease in the next 'patience' rounds, stop
                if not any(val_metric[i] > val_metric[i+j] for j in range(1, patience+1)):
                    early_stopping_round = i
                    break
        
        # If the loop completes without breaking, no early stopping is identified
        if early_stopping_round is None:
            early_stopping_round = len(val_metric) - 1
        
        # If the loop completes without breaking, no early stopping is identified
        if early_stopping_round is None:
            early_stopping_round = len(val_metric) - 1

        plt.figure(figsize=figsize)
        plt.plot(range(len(train_metric)), train_metric, color='#643EF0', label='Train', linewidth=2)
        plt.plot(range(len(val_metric)), val_metric, color='#FF2F5C', label=eval_set_name.capitalize(), linewidth=2)
        plt.axvline(x=early_stopping_round, color='grey', linestyle='--', label='Early Stopping',linewidth=2)
        plt.scatter(early_stopping_round, val_metric[early_stopping_round], s=100, color='#E1FF58', edgecolor='black', zorder=5)
        annotation_text =  f'ESP\n {early_stopping_round}'
        plt.annotate(annotation_text,
                (early_stopping_round, val_metric[early_stopping_round]),
                textcoords="offset points",
                xytext=(0,10),
                ha='center',
                fontsize=11,
                bbox=dict(boxstyle="round4", fc="w", edgecolor="white"))

        # Find the best metric value and round for the validation set
        if metric in ['auc', 'aucpr', 'f1']:  # Metrics where higher is better
            best_round = np.argmax(val_metric)
            best_metric_value = np.max(val_metric)
        else:  # Metrics like 'logloss' where lower is better
            best_round = np.argmin(val_metric)
            best_metric_value = np.min(val_metric)

        plt.scatter(best_round, best_metric_value, color='#9CF9FE', s=100, edgecolor='black', zorder=5)
        annotation_text = f"Best\n{metric.capitalize()}:\n{best_metric_value:.2f}"
        plt.annotate(annotation_text,
                     (best_round, best_metric_value),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center',
                     fontsize=10,
                     bbox=dict(pad=0.2, boxstyle="round4", fc="w", edgecolor="white"))

        plt.title(f'{model_name} {metric.capitalize()}', fontsize=16)
        plt.xlabel('Rounds', fontsize=14)
        plt.ylabel(metric.capitalize(), fontsize=14)
        plt.legend(loc='best', framealpha=1.0,fontsize='small')
        plt.grid(which='major', color='#DDDDDD', linewidth=0.6)
        plt.xticks(range(0, len(train_metric)+1, max(len(train_metric)//10, 1)))
        plt.tight_layout()

        if show_fig:
            plt.show()
        else:
            plt.savefig(f"{model_name}_trial_{trial_number}_{metric}_plot.png")
    except Exception as e:
        print(f"Unexpected error in plot_evaluation_results: {e}")



# ---------------------- MODEL COMPARISON: Check of y_vals of different models ---------------------- #

def compare_y_val_arrays(y_val_dict):
    """
    Compares y_val arrays from different models to ensure consistency in cross-validation splits.
    It checks if all y_val arrays have the same length and whether they are identical in terms of order and values.
    
    Parameters:
    - y_val_dict (dict): A dictionary where keys are model labels and values are the y_val arrays.
    
    Returns:
    - DataFrame: A summary DataFrame that includes model name, length of y_val array, order comparison,
                 unique values, their counts, and number of unique values.
    
    Raises:
    - ValueError: If y_val_dict is not a dictionary.
    - Exception: For other unexpected issues.
    """
    try:
        # Ensure input is a dictionary
        if not isinstance(y_val_dict, dict):
            raise ValueError("Input must be a dictionary with model labels as keys and y_val arrays as values.")

        labels = list(y_val_dict.keys())
        y_val_arrays = list(y_val_dict.values())
        
        # Check and log the lengths of y_val arrays
        logging.info("Checking lengths of y_val arrays...")
        lengths = [len(arr) for arr in y_val_arrays]
        if len(set(lengths)) > 1:
            logging.error("Not all y_val arrays have the same length.")
            return pd.DataFrame()  # Early return if lengths differ

        logging.info(f"All y_val arrays have the same length: {lengths[0]}")
        comparison_data = []
        
        # Initially assume all arrays are identical in order
        overall_identical_order = True

        # Base array for comparison
        base_array = y_val_arrays[0]
        base_unique_values, base_counts = np.unique(base_array, return_counts=True)

        # Compare with base array to check for order differences
        for i, arr in enumerate(y_val_arrays):
            unique_values, counts = np.unique(arr, return_counts=True)
            identical_order = "Identical" if np.array_equal(base_array, arr) else "NOT"
            if identical_order == "NOT":
                overall_identical_order = False  # Found at least one array not identical

            # Append comparison results to the list
            comparison_data.append({
                'y_val array': labels[i],
                'Length': lengths[i],
                'dUniques': unique_values.tolist(),
                'Distribution': counts.tolist(),
                'Order': identical_order,
            })

        # Create a DataFrame from the comparison data
        comparison_df = pd.DataFrame(comparison_data)
        
        # Log summary information based on comparison results
        if overall_identical_order:
            logging.info("All y_val arrays are identical across models.")
        else:
            logging.error("Differences found in order or values among y_val arrays.")

        print("\n")
        print(comparison_df)
        return comparison_df
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return pd.DataFrame()


# ----------------- MODEL PERFORMANCE SUMMARY -------------- #

def plot_performance_summary(model_metrics_dict):
    """
    Creates a DataFrame comparing different models and their metrics.
    
    Args:
    - model_metrics_dict (dict): A dictionary where keys are model names and values are DataFrames of metrics.
    
    Returns:
    - DataFrame: A DataFrame with the comparison of models and their metrics.
    """
    all_model_metrics = []
    
    # Iterate through each model and its metrics DataFrame
    for model_name, metrics_df in model_metrics_dict.items():
        # If the DataFrame uses 'Metric' as a column (AE_NN), pivot it to match others
        if 'Metric' in metrics_df.columns:
            metrics_df = metrics_df.set_index('Metric')
            # Rename columns to standard 'Validation Avg' if required
            if 'Test Set' in metrics_df.columns:
                metrics_df.rename(columns={'Test Set': 'Validation Avg'}, inplace=True)
        
        model_metrics = {'Model': model_name}
        for metric in ['precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'balanced_acc']:
            # Check if the metric exists in the DataFrame and add it to the dictionary
            if metric in metrics_df.index and 'Validation Avg' in metrics_df.columns:
                model_metrics[metric] = metrics_df.at[metric, 'Validation Avg']
            else:
                model_metrics[metric] = None  # Use None for metrics that are not available
        
        # Add the dictionary of metrics for this model to the list
        all_model_metrics.append(model_metrics)
    
    # Convert list of dictionaries into a DataFrame
    comparison_df = pd.DataFrame(all_model_metrics)

    # Optionally, sort the DataFrame based on a specific metric
    # Ensure 'AUC-ROC' column exists before sorting
    if 'pr_auc' in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by='pr_auc', ascending=False)
    
    print(comparison_df)



# -------------------- FINAL MODEL TRAINING: Threshold Optimization -------------------- #

def evaluate_threshold_metrics(y_true, model, X_val, chosen_metric='f1', start=0.1, end=1.0, step=0.1, verbose=True):
    """
    Evaluate and print classification metrics at specified thresholds for a given model and validation set,
    highlighting the best threshold for a chosen metric.

    Parameters:
    - y_true (array-like): True binary labels in the validation set.
    - model: Trained model object that must have either a predict_proba (for classifiers) or predict method.
    - X_val (array-like): Features of the validation set.
    - chosen_metric (str): The performance metric to optimize ('balanced_acc', 'precision', 'recall', 'f1').
    - start (float): Starting point of the threshold range to evaluate.
    - end (float): Ending point of the threshold range.
    - step (float): Step size to iterate through the threshold range.
    - verbose (bool): If True, prints detailed metrics for each threshold.
    """
    try:
        # Convert X_val (DataFrame) to DMatrix before prediction
        dval = xgb.DMatrix(X_val)

        # Check if model can predict probabilities, otherwise use predict method
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(dval)[:, 1]
        else:
            y_pred_proba = model.predict(dval)
            y_pred_proba = np.clip(y_pred_proba, 0, 1)  # Clip to ensure valid probability range

        # Validate that the lengths of the true labels and predicted probabilities match
        if len(y_true) != len(y_pred_proba):
            logging.error("Length of y_true and y_pred_proba must be the same.")
            raise ValueError("Length of y_true and y_pred_proba must be the same.")
        # Ensure all predicted probabilities are within the valid range [0, 1]
        if not all(0 <= prob <= 1 for prob in y_pred_proba):
            logging.error("y_pred_proba values must be between 0 and 1.")
            raise ValueError("y_pred_proba values must be between 0 and 1.")
    except Exception as e:
        logging.exception("An error occurred during model evaluation.")
        raise e

    # Calculate AUC-ROC to evaluate model's ability to discriminate between classes
    auc_roc = roc_auc_score(y_true, y_pred_proba)  #  threshold-independent

    # Calculate precision-recall pairs for different probability thresholds
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    # Compute the area under the precision-recall curve
    pr_auc = auc(recall, precision) # summary measure of precision & recall across all thresholds
    
    logging.info("AUC-ROC and PR AUC calculated successfully.")

    # Initialize a dictionary to store metrics calculated for each threshold
    metrics = {
        'threshold': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'pr_auc': [],  
        'auc_roc': auc_roc,  # Overall AUC-ROC for the model
        'balanced_acc': []  # Balanced Accuracy for each threshold
    }

    # Loop over thresholds, starting from 'start' to 'end' with steps of 'step'
    for threshold in np.arange(start, end + step, step):
        # Predict labels based on threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Append computed metrics to respective lists in the dictionary
        metrics['threshold'].append(threshold)
        metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_true, y_pred))
        metrics['f1'].append(f1_score(y_true, y_pred))
        metrics['balanced_acc'].append(balanced_accuracy_score(y_true, y_pred))

    # Find the index of the best metric according to the chosen optimization metric
    best_index = np.argmax(metrics[chosen_metric])
    # Extract the best threshold and its corresponding metric value
    best_threshold = metrics['threshold'][best_index]
    best_metric_value = metrics[chosen_metric][best_index]

    # Print the best threshold and its metric value
    print(f"\nBest threshold for {chosen_metric}: {best_threshold:.2f} = {best_metric_value:.4f}\n")
    # Print overall model performance metrics
    print(f"PR AUC : {pr_auc:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}\n")
    print('-' * 30)  

    # If verbose is True, print detailed metrics for each threshold
    if verbose:
        print(f"\nMetrics for each threshold from {start} to {end}:")
        for i, threshold in enumerate(metrics['threshold']):
            print(f"\nMetrics at threshold: {threshold:.2f}")
            print(f"precision : {metrics['precision'][i]:.4f}")
            print(f"recall    : {metrics['recall'][i]:.4f}")
            print(f"F1 Score  : {metrics['f1'][i]:.4f}")
            #print(f"PR AUC    : {metrics['pr_auc'][i]:.4f}")
            #print(f"AUC-ROC   : {metrics['auc_roc'][i]:.4f}")
            print(f"Bal. Acc  : {metrics['balanced_acc'][i]:.4f}\n")
            print('-' * 30)

    logging.info("Metrics calculated successfully across all thresholds.")


# ---------------------- FINAL MODEL: CREATE PREDICTIONS ON TEST DATA ---------------------- #

def predict_with_model(df_test, model, Identifier=None, threshold=0.5):
    """
    Use a trained model without a target to predict outcomes on test data. This function is designed
    to be robust and flexible, handling various model types and accommodating scenarios where
    identifiers may or may not be present.

    Parameters:
    - df_test (pd.DataFrame): DataFrame containing the test data features.
    - model (ML model): Trained machine learning model.
    - Identifier (str or list of str, optional): Column name(s) used as a unique ID for each entry.
    - threshold (float, optional): Threshold for converting probabilities to binary output.

    Returns:
    - pd.DataFrame: DataFrame containing the Identifier, predicted probabilities, and predicted classes.
    """
    try:
        # Prepare test data by excluding Identifier from features if specified
        if Identifier is not None:
            identifier_col = Identifier[0] if isinstance(Identifier, list) else Identifier
            X_test = df_test.drop(Identifier, axis=1, errors='ignore')
        else:
            X_test = df_test.copy()
            identifier_col = 'index'  # Default identifier if none provided
            df_test[identifier_col] = df_test.index
            logging.info("No Identifier specified, using entire DataFrame for predictions.")

        # Convert DataFrame to DMatrix if the model is an XGBoost Booster
        if isinstance(model, xgb.Booster):
            X_test = xgb.DMatrix(X_test)
            #logging.info("Converted DataFrame to DMatrix for XGBoost prediction.")

        # Generate predictions
        y_test_pred_prob = model.predict(X_test)
        y_test_pred = (y_test_pred_prob >= threshold).astype(int)
        #logging.info("Generated predictions with the model.")

        # Creating the predictions DataFrame with dynamic identifier column name
        predictions_df = pd.DataFrame({
            identifier_col: df_test[identifier_col],
            'Prediction_Probability': y_test_pred_prob,
            'Predicted_Class': y_test_pred
        })
        logging.info("Created predictions for Test Data successfully.\n")
        
        #print(predictions_df.head())
        
        return predictions_df

    except Exception as e:
        logging.error("Error in predict_with_model: {}".format(e), exc_info=True)
        raise

# ----------------- FINAL MODEL: Comparison and Evaluation with actual predictions -------------- #

def evaluate_predictions(df_predictions_dict, df_actuals, 
                         Identifier, 
                         target_column='Attrition'):
    """
    Evaluate prediction performance by merging predicted results with actual labels for multiple models,
    calculating various performance metrics such as F1 score, precision-recall AUC, ROC AUC, and balanced accuracy.

    Parameters:
    - df_predictions_dict (dict): Dictionary containing model names as keys and DataFrames with predictions as values.
    - df_actuals (pd.DataFrame): DataFrame containing actual labels and identifiers.
    - Identifier (str): Column name used as the identifier to merge predictions with actuals.
    - target_column (str): Column name of the target variable in `df_actuals`. Defaults to 'Attrition'.

    Returns:
    - pd.DataFrame: Summary DataFrame with evaluation metrics sorted by F1 score in descending order.

    Raises:
    - ValueError: If the identifier or target column is missing from any DataFrame.
    """
    try:
        # Ensure column names are stripped of any leading/trailing spaces
        df_actuals.columns = df_actuals.columns.str.strip()
        
        results = []
        for model_name, df_predictions in df_predictions_dict.items():
            df_predictions.columns = df_predictions.columns.str.strip()

            identifier_column_predictions = 'Identifier'

            # Rename the identifier column in predictions to match the actuals DataFrame
            df_predictions.rename(columns={identifier_column_predictions: Identifier}, inplace=True)

            if Identifier not in df_predictions.columns or Identifier not in df_actuals.columns:
                raise ValueError(f"'{Identifier}' is missing in the provided DataFrames. Available columns in predictions: {df_predictions.columns}, in actuals: {df_actuals.columns}")
            if target_column not in df_actuals.columns:
                raise ValueError(f"'{target_column}' is missing in the actuals DataFrame. Available columns: {df_actuals.columns}")

            merged_df = df_predictions.merge(df_actuals[[Identifier, target_column]], on=Identifier)

            # Calculate correct predictions for true positives and true negatives
            true_positives = ((merged_df['Predicted_Class'] == 1) & (merged_df[target_column] == 1)).sum()
            true_negatives = ((merged_df['Predicted_Class'] == 0) & (merged_df[target_column] == 0)).sum()

            # Prepare true labels and predicted labels/probabilities for scoring
            y_true = merged_df[target_column]
            y_pred = merged_df['Predicted_Class']
            y_pred_prob = merged_df['Prediction_Probability']

            # Compute evaluation metrics
            f1 = f1_score(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_pred_prob)
            roc_auc = roc_auc_score(y_true, y_pred_prob)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)

            results.append({
                "Model |": f"{model_name} |",
                "Actual (total) |": f"TP:{merged_df[target_column].sum()}, TN:{len(merged_df) - merged_df[target_column].sum()} |",
                "Predicted |": f"P: {y_pred.sum()}, N: {len(y_pred) - y_pred.sum()} |",
                "Correct Predictions |": f"TP: {true_positives}, TN: {true_negatives} |",
                "F1 |": f"{f1:.2f} |",
                "PR_AUC |": f"{pr_auc:.2f} |",
                "ROC_AUC |": f"{roc_auc:.2f} |",
                "Balanced_Acc |": f"{balanced_acc:.2f} |"
            })

        # Create DataFrame from results and sort by F1 score in descending order
        summary_df = pd.DataFrame(results)
        df = summary_df.sort_values(by='F1 |', ascending=False)

        print(df)
        return df

    except Exception as e:
        print("An error occurred during the evaluation of predictions with actuals: ", e)
        raise


# ----------------- FINAL MODEL: Create predictions & probabilities -------------- #

def create_predictions(predictions_df, 
                       top_n=None, 
                       model_name=None):
    """
    Processes and displays predictions from a specified model, providing insights into predicted classes.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing the prediction results.
        top_n (int, optional): The number of top predictions to display. Defaults to 10 if None.
        model_name (str, optional): Name of the model for reference in logs and outputs.

    Returns:
        pd.DataFrame: A subset of the predictions DataFrame, limited to the top_n items.
    
    Raises:
        ValueError: If the input is not a DataFrame or lacks the required 'Predicted_Class' column.
    """
    try:
        if not isinstance(predictions_df, pd.DataFrame):
            logging.error("Invalid input type: Expected a pandas DataFrame.")
            raise ValueError("Input is not a DataFrame. Please provide a DataFrame.")

        if 'Predicted_Class' not in predictions_df.columns:
            logging.error("Missing necessary column in DataFrame: 'Predicted_Class'")
            raise ValueError("'Predicted_Class' column is necessary but was not found in the DataFrame.")
        
        display_rows = top_n or 10
        
        sorted_predictions = predictions_df.sort_values(by='Prediction_Probability', ascending=False)
        
        logging.info(f"Displaying top {display_rows} predictions for {model_name}:")
        print('\n' + sorted_predictions.head(display_rows).to_string())

        test_pred_counts = sorted_predictions['Predicted_Class'].value_counts()
        print('\n' + test_pred_counts.to_string())

        pos_class_count = sorted_predictions[
            (sorted_predictions['Prediction_Probability'] > 0.7) & 
            (sorted_predictions['Predicted_Class'] == 1)
        ].shape[0]

        print(f"\n{model_name}: Number of pos. labeled over >70% probability: {pos_class_count}")

        #return sorted_predictions.head(display_rows)
    
    except Exception as e:
        logging.error("An error occurred while creating predictions.", exc_info=True)
        raise e
    

# -------------------- FINAL MODEL TRAINING: Align Features of Test DF -------------------- #

def align_features(df_train, df_test):
    """
    Aligns the columns of df_test to match the columns in df_train. 
    Columns in df_test not found in df_train are dropped, and columns in df_train 
    not found in df_test are reported and skipped.
    
    Parameters:
    - df_train (pd.DataFrame): The reference DataFrame with the desired column structure.
    - df_test (pd.DataFrame): The DataFrame to be aligned to df_train.
    
    Returns:
    - pd.DataFrame: A new DataFrame based on df_test but with columns aligned to df_train.
    
    Raises:
    - ValueError: If either df_train or df_test is not a pandas DataFrame.
    """
    
    # Ensure input is of type pd.DataFrame
    if not isinstance(df_train, pd.DataFrame) or not isinstance(df_test, pd.DataFrame):
        logging.error("Input is not a pandas DataFrame.")
        raise ValueError("Both df_train and df_test must be pandas DataFrame instances.")
    
    logging.info("Starting feature alignment process.")
    
    # Identify columns to keep (columns in both df_train and df_test)
    cols_to_keep = [col for col in df_train.columns if col in df_test.columns]
    
    # Identify missing columns in df_test that are present in df_train
    missing_cols = [col for col in df_train.columns if col not in df_test.columns]
    
    # Log missing columns, if any
    if missing_cols:
        logging.info(f"Columns present in df_train but missing in df_test: {missing_cols}")
    
    # Create a new DataFrame with aligned columns
    df_test_aligned = df_test[cols_to_keep].copy()
    
    logging.info("Feature alignment completed successfully.\n")
    print(f"Train Df: {df_train.shape}")
    print(f"Test  Df: {df_test_aligned.shape}")
    return df_test_aligned
