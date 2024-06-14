# pre_configure.py

# ---------------------- Standard Library Imports ---------------------- #
import logging
import os
import sys
import warnings

# Define the log file path
log_file_path = 'pre_configure_log.txt'

# Set up the log file handler
log_file_handler = logging.FileHandler(log_file_path, mode='w')
log_file_handler.setLevel(logging.INFO)
log_file_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))

# Remove all handlers associated with the root logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up basic configuration for logging
logging.basicConfig(handlers=[log_file_handler, logging.StreamHandler(sys.stdout)],
                    level=logging.INFO,
                    format='%(levelname)s:%(name)s: %(message)s')

# Create a logger
logger = logging.getLogger()

# Set TensorFlow log level to suppress messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Redirect TensorFlow logs to /dev/null
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)
tf_logger.handlers = [logging.NullHandler()]

# ---------------------- Data Handling and Processing ---------------------- #
import numpy as np
import pandas as pd

# ---------------------- Data Visualization ---------------------- #
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Machine Learning Models and Algorithms ---------------------- #
import lightgbm as lgb
import xgboost as xgb
from sklearn.utils import class_weight

# ---------------------- Data Preprocessing and Feature Engineering ---------------------- #
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import MinMaxScaler

# ---------------------- Model Training and Evaluation ---------------------- #
from sklearn.model_selection import (StratifiedKFold, KFold, RepeatedStratifiedKFold, train_test_split)

# ---------------------- Neural Network Specifics ---------------------- #
from tensorflow.keras import layers, models, callbacks, regularizers, initializers, optimizers
from keras.layers import Dropout, BatchNormalization, LeakyReLU
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

# ---------------------- Hyperparameter Optimization ---------------------- #
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import HyperbandPruner, MedianPruner, PatientPruner

# ---------------------- Configuration Settings ---------------------- #
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter('ignore')

# Apply custom CSS to change output background to white and text to black in Jupyter
from IPython.display import display, HTML
display(HTML('''
    <style>
    .output_area pre {
        background-color: white !important;
        color: black !important;
    }
    </style>
'''))

# Disable GPU usage for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


