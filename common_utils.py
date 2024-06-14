# ---------------------- Execute save dataframes function & print results ---------------------- #
# ---------------------- Standard Library Imports ---------------------- #
import json
import logging
import os
import json
import re
import sys
import warnings
from fractions import Fraction
import datetime
from datetime import datetime
import unicodedata
from dateutil.parser import parse
from dateutil import parser as date_parser
from typing import Dict, List, Tuple, Union, Any
import ipywidgets as widgets
from IPython.display import display, Markdown, HTML, Image
import importlib

# ---------------------- Data Handling and Processing ---------------------- #
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------- Data Visualization ---------------------- #
import matplotlib.pyplot as plt
from plotly.offline import iplot, plot, init_notebook_mode
import seaborn as sns

# Initialize Plotly to run offline in a Jupyter Notebook
init_notebook_mode(connected=True)

# ---------------------- Machine Learning Models and Algorithms ---------------------- #
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier

# ---------------------- Data Preprocessing and Feature Engineering ---------------------- #
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, OneHotEncoder, RobustScaler, PolynomialFeatures, StandardScaler
)

# ---------------------- Model Training and Evaluation ---------------------- #
from sklearn.model_selection import (train_test_split)

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

# ---------------------- IMPORT MY_MODULES  ---------------------- #
import model_utils as mu
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
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# ---------------------- CHECK ENVIRONMENT ---------------------- #
class EnvironmentChecker:
    """
    Check the execution environment
    ================================

    A class to check and report the execution environment (interactive or non-interactive).

    Attributes:
    -----------
        interactive (bool): Determines if the script is running in an interactive environment.

    Methods:
    --------
        is_interactive(): Returns True if the environment is interactive.
        log_environment(): Logs the environment type to a logging service.
        print_environment(): Prints the environment type to the standard output.
    """
    def __init__(self, show=True):
        # Determine the interactive state once during instantiation
        self.interactive = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules
        self.show = show
        
        # Automatically log and print the environment based on the show flag
        if self.show:
            self.log_environment()
            #self.print_environment()

    def is_interactive(self):
        """Return True if code is running in an interactive environment."""
        return self.interactive

    def log_environment(self):
        """Log the current operating environment using the logging library."""
        if self.is_interactive():
            logging.info("Running in an interactive environment (e.g., Jupyter).")
        else:
            logging.info("Running in a non-interactive environment (e.g., standard Python script).")

    def print_environment(self):
        """Print the current operating environment to the standard output."""
        if self.is_interactive():
            print("This is an interactive environment.")
        else:
            print("This is a non-interactive environment.")

# ---------------------- LOAD WIDGET ---------------------- #
def load_widget_state(base_path=None, filename='Saved_Dataframes/dd_widget.json'):
    """
    Load the widget state from a JSON file and display the dropdown widget.

    Parameters:
    -----------
    base_path : str
        The base path where the JSON file is located.
    filename : str
        The path to the JSON file from which the state will be loaded.
    """
    if base_path is None:
        base_path = os.getenv('BASE_PATH')
    
    full_path = os.path.join(base_path, filename)
    
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            state = json.load(f)
        
        # Create the dropdown widget with the loaded state
        dropdown_widget = widgets.Dropdown(
            options=state['options'],
            value=state['value'],
            description='Select:',
            disabled=False,
        )

        doc_output = widgets.Output()

        # Define the function to display the documentation based on the selected value
        def display_doc(change):
            if change['new'] != 'Select function or class':
                doc_output.clear_output()
                with doc_output:
                    doc_content = f"### {change['new']} Documentation\n\nDocumentation content here."
                    display(Markdown(doc_content))

        dropdown_widget.observe(display_doc, names='value')
        display(dropdown_widget, doc_output)
    else:
        print(f"No saved widget state found at {full_path}.")


# ---------------------- SAVE & LOAD CLASS: DATAFRAME MANAGER ---------------------- #

class DataFrameManager:
    """
    DataFrame Management for Saving and Loading
    ============================================

    This class provides robust functionality for managing the saving and loading of pandas DataFrames in various formats,
    including Parquet, CSV, and Excel. It is designed to ensure the integrity and datatype consistency of the data 
    throughout the save and load processes. Additionally, it supports creating necessary directories if they do not exist 
    and handles the saving and loading of DataFrame metadata, such as column datatypes.

    The class saves an accompanying JSON file that stores the datatype information of each column in the DataFrame.
    This ensures that when the DataFrame is reloaded, the original datatypes are preserved, preventing potential issues
    with data type conversions that could arise from loading and saving operations.

    Supported Formats:
    ------------------
    - Parquet (`.parquet`)
    - CSV (`.csv`)
    - Excel (`.xlsx`)

    Parameters:
    -----------
    - base_path (str, optional): The root directory where the DataFrame will be stored. If not specified,
                                 the function will attempt to retrieve the base path from the `BASE_PATH`
                                 environment variable.

    Attributes:
    -----------
    - base_path (str): The base directory for all file operations. If not provided, it tries to fetch from environment variables.

    Methods:
    --------
    - save_df(df, df_name, directory, format, max_rows): Saves a DataFrame to a file with the specified format.
    - load_data(file_name, directory, **kwargs): Loads a DataFrame from a file with the specified format.

    Raises:
    -------
    - ValueError: If no base path is provided or set in the environment variables.

    Note:
    -----
    - This class assumes that the input DataFrame has already been cleaned and preprocessed.
    - The `base_path` attribute must be set either through the parameter or the `BASE_PATH` environment variable.

    Usage:
    ------
    >>> manager = DataFrameManager(base_path="/path/to/save")
    >>> df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    >>> manager.save_df(df, df_name="example", directory="data", format="csv", max_rows=100)
    >>> loaded_df = manager.load_data("example.csv", directory="data")
    >>> print(loaded_df)

    Detailed Explanation:
    ---------------------
    save_df Method:
    ---------------
    Saves a dataframe in specified format (Parquet, CSV, or Excel).

    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame to save.
    - df_name (str, default 'dataframe'): The base name of the file to which the DataFrame will be saved, without an extension.
    - directory (str, default 'SavedData'): The subdirectory under the base path where the file will be saved.
    - format (str, default 'parquet'): The file format for saving the DataFrame. Options are 'parquet', 'csv', or 'excel'.
                                       Each format corresponds to the respective pandas function (`to_parquet`, `to_csv`, `to_excel`).
    - max_rows (int, optional): The maximum number of rows to save in the file. If None, all rows are saved.

    Raises:
    -------
    - ValueError: If `base_path` is not set or the specified file format is not supported.
    - Exception: Propagates any exceptions raised during file saving, such as I/O errors or pandas-related errors.

    Returns:
    --------
    - None: Function does not return a value but saves the file to disk.

    load_data Method:
    -----------------
    Load data from a specified file within a base directory and subdirectory, supporting CSV, Excel, and Parquet formats. 
    Includes extensive error handling for missing files and directories.

    Parameters:
    -----------
    - file_name (str): Name of the file to load. The function supports 'csv', 'xlsx', and 'parquet' files.
    - directory (str): Subdirectory within the base path where the file is located.
    - **kwargs (dict): Additional keyword arguments that are passed to the pandas read function
                       (e.g., `pd.read_csv`, `pd.read_excel`, or `pd.read_parquet`).

    Raises:
    -------
    - ValueError: If `base_path` is not provided and not set in the environment variables, or if the file format is not supported.

    Returns:
    --------
    - pd.DataFrame or None: Returns a pandas DataFrame if the file is successfully loaded; otherwise, None if the file does not exist,
                            the directory is not found, or an error occurs during loading.

    Example:
    --------
    ```python
    import pandas as pd

    # Initialize the DataFrameManager
    manager = DataFrameManager(base_path="/path/to/save")

    # Sample DataFrame
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6]
    })

    # Save DataFrame
    manager.save_df(df, df_name="example", directory="data", format="csv", max_rows=100)

    # Load DataFrame
    loaded_df = manager.load_data("example.csv", directory="data")
    print(loaded_df)
    ```
    """
    def __init__(self, base_path=None):
        self.base_path = base_path or os.getenv('BASE_PATH')
        if self.base_path is None:
            raise ValueError("Base path must be provided or set in environment variables.")

    def save_df(self, df, df_name='dataframe', directory='SavedData', format='parquet', max_rows=None):
        """
        Saves a dataframe in specified format (Parquet or CSV).

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to save.
        df_name : str, default 'dataframe'
            The base name of the file to which the DataFrame will be saved, without an extension.
        directory : str, default 'SavedData'
            The subdirectory under the base path where the file will be saved.
        base_path : str, optional
            The root directory where the DataFrame will be stored. If not specified,
            the function will attempt to retrieve the base path from the `BASE_PATH`
            environment variable.
        format : str, default 'parquet'
            The file format for saving the DataFrame. Options are 'parquet', 'csv', or 'excel'.
            Each format corresponds to the respective pandas function (`to_parquet`, `to_csv`, `to_excel`).
        max_rows : int, optional
            The maximum number of rows to save in the file. If None, all rows are saved.

        Raises
        ------
        ValueError
            If `base_path` is not set or the specified file format is not supported.
        Exception
            Propagates any exceptions raised during file saving, such as I/O errors or pandas-related errors.

        Returns
        -------
        None
            Function does not return a value but saves the file to disk.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected df to be a pandas DataFrame, but got {type(df).__name__}")

        try:
            # Limit the dataframe to max_rows if specified
            if max_rows is not None:
                df = df.head(max_rows)

            # Create a new folder within the base path if it does not already exist
            save_folder = os.path.join(self.base_path, directory)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)
                logging.info("Created directory for saving dataframes.")

            # Define file extension based on format
            file_extension = format
            if format == 'excel':
                file_extension = 'xlsx'  # Correct extension for Excel files

            # Define file name with the correct extension
            df_filename = f"{df_name}.{file_extension}"

            # Define the full file path without the extension for use in saving dtypes
            file_path = os.path.join(save_folder, df_name)

            # Save the dataframe in the specified format
            if format == 'parquet':
                df.to_parquet(os.path.join(save_folder, df_filename))
            elif format == 'csv':
                df.to_csv(os.path.join(save_folder, df_filename), index=False)
            elif format == 'excel':
                df.to_excel(os.path.join(save_folder, df_filename), index=False, engine='openpyxl')
            else:
                raise ValueError("Unsupported file format. Choose 'parquet', 'csv', or 'excel'.")

            logging.info(f"'{df_filename}' saved successfully.")

            # Save the DataFrame's dtype information
            dtype_path = f"{os.path.splitext(file_path)[0]}_dtypes.json"
            dtypes = df.dtypes.apply(lambda x: str(x)).to_dict()
            # dtypes = {col: str(dtype) for col, dtype in df.dtypes.iteritems()}
            with open(dtype_path, 'w') as f:
                json.dump(dtypes, f)
            logging.info(f"Data types JSON saved successfully at {dtype_path}\n")

        except Exception as e:
            logging.error(f"Failed to save dataframe: {e}\n")
            raise

    def load_data(self, file_name, directory, **kwargs):
        """
        Load data from a specified file within a base directory and subdirectory, 
        supporting CSV, Excel, and Parquet formats. Includes extensive error handling for 
        missing files and directories.

        Parameters
        ----------
        file_name : str
            Name of the file to load. The function supports 'csv', 'xlsx', and 'parquet' files.
        directory : str
            Subdirectory within the base path where the file is located.
        base_path : str, optional
            The base path of the dataset directory. If not specified, the function will
            attempt to retrieve it from the `BASE_PATH` environment variable.
        **kwargs : dict
            Additional keyword arguments that are passed to the pandas read function
            (e.g., `pd.read_csv`, `pd.read_excel`, or `pd.read_parquet`).

        Raises
        ------
        ValueError
            If `base_path` is not provided and not set in the environment variables, or if the
            file format is not supported.

        Returns
        -------
        pandas.DataFrame or None
            Returns a pandas DataFrame if the file is successfully loaded; otherwise, None if
            the file does not exist, the directory is not found, or an error occurs during loading.
        """    
        full_path = os.path.join(self.base_path, directory, file_name)

        # Check if the directory exists
        if not os.path.exists(os.path.join(self.base_path, directory)):
            logging.warning(f"Directory not found in: {os.path.join(self.base_path, directory)}")
            return None

        # Check if the file exists
        if not os.path.isfile(full_path):
            logging.warning(f"File not found in: {full_path}")
            return None

        try:
            # Determine the file format and load the DataFrame
            if file_name.endswith('.csv'):
                data = pd.read_csv(full_path, **kwargs)
            elif file_name.endswith('.xlsx'):
                kwargs.pop('low_memory', None)  # Excel doesn't support 'low_memory'
                data = pd.read_excel(full_path, **kwargs)
            elif file_name.endswith('.parquet'):
                data = pd.read_parquet(full_path, **kwargs)
            else:
                raise ValueError("Unsupported file format. Please use CSV, Excel, or Parquet.")

            # In the load_data method, before attempting to load the JSON:
            dtype_base_path = full_path.rsplit('.', 1)[0]  # Removes last extension
            dtype_path = f"{dtype_base_path}_dtypes.json"
            #print(f"Attempting to load JSON dtypes from: {dtype_path}")
            if os.path.exists(dtype_path):
                with open(dtype_path, 'r') as f:
                    dtypes = json.load(f)
                for col, dtype in dtypes.items():
                    data[col] = data[col].astype(dtype)
                #logging.info("JSON dtype file found. Data types restored.")
            else:
                logging.info("No JSON dtype file found. Data types might not be fully restored.")

            logging.info(f"'{file_name}' loaded successfully from {full_path}, \n -> shape: {data.shape}\n")
            return data

        except Exception as e:
            logging.error(f"Failed to load data from {full_path}: {e}")
            return None

# ---------------------- Function to LOAD DATA & DF ---------------------- #

def load_data(file_name, directory, base_path=None, **kwargs):
    """
    Load data from a specified file within a base directory and subdirectory, 
    supporting CSV, Excel, and Parquet formats. Includes extensive error handling for 
    missing files and directories.

    Parameters
    ----------
    file_name : str
        Name of the file to load. The function supports 'csv', 'xlsx', and 'parquet' files.
    directory : str
        Subdirectory within the base path where the file is located.
    base_path : str, optional
        The base path of the dataset directory. If not specified, the function will
        attempt to retrieve it from the `BASE_PATH` environment variable.
    **kwargs : dict
        Additional keyword arguments that are passed to the pandas read function
        (e.g., `pd.read_csv`, `pd.read_excel`, or `pd.read_parquet`).

    Raises
    ------
    ValueError
        If `base_path` is not provided and not set in the environment variables, or if the
        file format is not supported.

    Returns
    -------
    pandas.DataFrame or None
        Returns a pandas DataFrame if the file is successfully loaded; otherwise, None if
        the file does not exist, the directory is not found, or an error occurs during loading.

    Examples
    --------
    >>> df = load_data('data.csv', 'my_data', '/path/to/data', sep=',')
    This would attempt to load a CSV file located at '/path/to/data/my_data/data.csv' with a comma separator.

    >>> df = load_data('dataset.xlsx', '2021/reports', index_col=0)
    This would load an Excel file using the first column as the index.
    """
    if base_path is None:
        base_path = os.getenv('BASE_PATH')
    if base_path is None:
        raise ValueError("base_path is not provided and not set in the environment variables.")
    
    full_path = os.path.join(base_path, directory, file_name)

    # Check if the directory exists
    if not os.path.exists(os.path.join(base_path, directory)):
        logging.warning(f"Directory not found in: {os.path.join(base_path, directory)}")
        return None

    # Check if the file exists
    if not os.path.isfile(full_path):
        logging.warning(f"File not found in: {full_path}")
        return None

    try:
        # Determine file format
        if file_name.endswith('.csv'):
            data = pd.read_csv(full_path, **kwargs)
        elif file_name.endswith('.xlsx'):
            kwargs.pop('low_memory', None)
            data = pd.read_excel(full_path, **kwargs)
        elif file_name.endswith('.parquet'):
            data = pd.read_parquet(full_path, **kwargs)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Parquet.")

        logging.info(f"Data loaded successfully from {full_path}")
        return data

    except Exception as e:
        logging.error(f"Failed to load data from {full_path}: {e}")
        return None

# ---------------------- Function to SAVE VERSIONS OF DATAFRAMES ---------------------- #

def save_df(df, df_name='dataframe', directory='SavedData', base_path=None, 
            format='parquet', max_rows=None):
    """
    Saves a dataframe in specified format (Parquet or CSV).

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to save.
    df_name : str, default 'dataframe'
        The base name of the file to which the DataFrame will be saved, without an extension.
    directory : str, default 'SavedData'
        The subdirectory under the base path where the file will be saved.
    base_path : str, optional
        The root directory where the DataFrame will be stored. If not specified,
        the function will attempt to retrieve the base path from the `BASE_PATH`
        environment variable.
    format : str, default 'parquet'
        The file format for saving the DataFrame. Options are 'parquet', 'csv', or 'excel'.
        Each format corresponds to the respective pandas function (`to_parquet`, `to_csv`, `to_excel`).
    max_rows : int, optional
        The maximum number of rows to save in the file. If None, all rows are saved.

    Raises
    ------
    ValueError
        If `base_path` is not set or the specified file format is not supported.
    Exception
        Propagates any exceptions raised during file saving, such as I/O errors or pandas-related errors.

    Returns
    -------
    None
        Function does not return a value but saves the file to disk.

    Examples
    --------
    >>> save_df(my_dataframe, df_name='example_data', directory='exports', format='csv')
    Saves `my_dataframe` as a CSV file in the `exports` directory under the current base path.
    """
    # Fetch the base path from environment if not provided
    if base_path is None:
        base_path = os.getenv('BASE_PATH')
        if base_path is None:
            raise ValueError("base_path is not provided and not set in the environment variables.")
        
    try:
        # Limit the dataframe to max_rows if specified
        if max_rows is not None:
            df = df.head(max_rows)

        # Create a new folder within the base path
        save_folder = os.path.join(base_path, directory)
        os.makedirs(save_folder, exist_ok=True)
        logging.info("Created directory for saving dataframes.")

        # Define file extension based on format
        file_extension = format
        if format == 'excel':
            file_extension = 'xlsx'  # Correct extension for Excel files

        # Define file name with the correct extension
        df_filename = f"{df_name}.{file_extension}"

        # Save the dataframe in the specified format
        if format == 'parquet':
            df.to_parquet(os.path.join(save_folder, df_filename))
        elif format == 'csv':
            df.to_csv(os.path.join(save_folder, df_filename), index=False)
        elif format == 'excel':
            # Ensure openpyxl is installed for xlsx format
            df.to_excel(os.path.join(save_folder, df_filename), index=False, engine='openpyxl')
        else:
            raise ValueError("Unsupported file format. Choose 'parquet', 'csv', or 'excel'.")

        logging.info(f"DataFrame saved successfully as {df_filename}")

    except Exception as e:
        logging.error(f"Failed to save dataframe: {e}")
        raise

    logging.info("DataFrame saved successfully.")


# ----------------- PREPROCESSING: Merging multiple DataFrames ----------------- #

def merge_datasets(datasets, dataset_names, join='inner', on=None, suffixes=('_x', '_y')):
    """
    Merge Dataframes
    ================

    Merges two or more datasets based on a specified join type and key. This function includes extensive error handling,
    detailed logging of the merging process, and outputs the dimensions of the datasets before and after merging.
    It is designed to be robust in handling common merging issues such as non-matching keys and varying row counts.

    Parameters:
    -----------
    - datasets (list of pd.DataFrame): List of DataFrame objects to be merged.
    - dataset_names (list of str): Names corresponding to the datasets, used for logging purposes.
    - join (str): Type of join to perform ('inner', 'outer', 'left', 'right'). Default is 'inner'.
    - on (str): The key column on which to join the DataFrames. All DataFrames must include this column.
    - suffixes (tuple): Suffixes to apply to overlapping column names in the different DataFrames. Default is ('_x', '_y').

    Returns:
    --------
    - pd.DataFrame or None: Returns a merged DataFrame if successful, None if an error occurs.

    Raises:
    -------
    - ValueError: If the join type is not recognized.
    - Exception: Propagates exceptions that indicate issues with DataFrame operations.

    Examples:
    ---------
    - Merging two customer data tables with an outer join to ensure no data is lost:
      >>> df_combined = merge_datasets([df1, df2], ['Table1', 'Table2'], join='outer', on='customer_id')
    """
    try:
        # Check that all elements in datasets are pandas DataFrames and the names list matches in length
        if not all(isinstance(ds, pd.DataFrame) for ds in datasets) or len(datasets) != len(dataset_names):
            print("Error: Inputs must be pandas DataFrame objects and names must match datasets count")
            return None

        # Ensure the merge key exists in all DataFrames
        if on is not None:
            missing_columns = [name for ds, name in zip(datasets, dataset_names) if on not in ds.columns]
            if missing_columns:
                print(f"Error: Merge key '{on}' not found in dataset(s): {', '.join(missing_columns)}")
                return None

        # Log the shape of each dataset before merging
        for ds, name in zip(datasets, dataset_names):
            print(f"{name} shape before merging: {ds.shape}")

        # Confirm the join type is valid
        if join not in ['inner', 'outer', 'left', 'right']:
            raise ValueError("Join type must be 'inner', 'outer', 'left', or 'right'")

        # Explanation of the join type and its implications
        join_info = {
            'inner': "Combines rows with matching keys in both datasets. Rows without matches are excluded.",
            'outer': "Includes all rows from all datasets. Missing matches are filled with NaNs.",
            'left': f"Includes all rows from {dataset_names[0]} and matching rows from other datasets. Rows in {dataset_names[0]} without matches in others will have NaNs for missing columns.",
            'right': f"Includes all rows from {dataset_names[-1]} and matching rows from other datasets. Rows in {dataset_names[-1]} without matches in others will have NaNs for missing columns."
        }
        print(f"Merging with a {join} join: {join_info[join]}")

        # Perform the merge
        merged_df = datasets[0]
        for i in range(1, len(datasets)):
            merged_df = pd.merge(merged_df, datasets[i], how=join, on=on, suffixes=suffixes)

        # Output the shape of the merged DataFrame
        print(f"\nMerged dataset shape now: {merged_df.shape}")
        return merged_df

    except Exception as e:
        print(f"An error occurred during merging: {e}")
        raise


# -------------------- ANALYSIS: Function to analyse data ------------------- #

def analyze_dataframe(df, dataset_name):
    """
    Analyze a Dataframe
    ===================

    Analyze a DataFrame to provide its shape, data types, non-null counts, and more.

    Args:
    -----
    df (DataFrame): The pandas DataFrame to analyze.

    Returns:
    --------
    tuple: A DataFrame containing column information and a string with shape information.
    """
    try:
        # DataFrame for storing column-wise analysis
        col_analysis = pd.DataFrame(df.dtypes, columns=['Data Type'])
        col_analysis['Non-Null Count'] = df.notnull().sum()
        col_analysis['Null Count'] = df.isnull().sum()
        col_analysis['NaN Count'] = df.isna().sum()

        # Calculate the percentage of NaNs and format it as string with two decimals
        col_analysis['Percentage of NaNs'] = col_analysis['NaN Count'] / df.shape[0] * 100
        col_analysis['Percentage of NaNs'] = col_analysis['Percentage of NaNs'].apply(lambda x: f"{x:.2f}")

        # Shape information as a string
        shape_info = (
            "-"*55 + "\n"
            f'Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}'
        )

        # Logging successful analysis
        logging.info("DataFrame analysis completed successfully.")

        # Printing the results in the desired format
        print(f"\nAnalysis of {dataset_name} Dataset:")
        print(col_analysis.to_string())
        print(shape_info)

    except Exception as e:
        logging.error(f"Error in analyze_dataframe: {e}")
        return None, None


# ---------------------- ANALYSIS: Function to compare dataframes ---------------------- #

def compare_dataframes(dfs):
    """
    Compares two pandas DataFrames based on their column names (case insensitive) and prints a summary of matched and unmatched columns.
    Returns a single DataFrame listing all columns from both DataFrames with a match indicator.

    Parameters:
    dfs (dict): A dictionary with two entries where keys are dataframe names and values are pandas DataFrames.

    Returns:
    pandas.DataFrame: DataFrame with columns from both DataFrames and a match indicator.

    Raises:
    ValueError: If not exactly two DataFrames are provided.
    TypeError: If the provided values are not pandas DataFrames.
    """
    
    # Validate input dictionary
    if len(dfs) != 2:
        raise ValueError("Input must be a dictionary with exactly two DataFrames")

    # Extract the dataframes and their names
    names = list(dfs.keys())
    df1_name, df2_name = names[0], names[1]
    df1, df2 = dfs[df1_name], dfs[df2_name]

    # Validate that both values are DataFrames
    if not all(isinstance(df, pd.DataFrame) for df in [df1, df2]):
        raise TypeError("All values in the input dictionary must be pandas DataFrames")

    # Normalize column names to lowercase for comparison
    df1_cols = df1.columns.str.lower()
    df2_cols = df2.columns.str.lower()

    # Create a dictionary to hold the comparison results
    comparison_dict = {}

    for col in df1_cols:
        match_col = df2.columns[df2_cols == col].tolist()[0] if col in df2_cols else None
        comparison_dict[col] = ("Yes" if match_col else "No")

    # Creating the comparison DataFrame
    comparison_df = pd.DataFrame({
        df1_name: df1.columns,  # Use original case for column names
        df2_name: [df2.columns[df2_cols == col].tolist()[0] if col in df2_cols else "-" for col in df1_cols],
        " Matching_col": [comparison_dict[col.lower()] for col in df1_cols]
    })

    # Handle columns in df2 not in df1
    unmatched_cols = set(df2_cols) - set(df1_cols)
    additional_rows = pd.DataFrame({
        df1_name: ["-"] * len(unmatched_cols),
        df2_name: [df2.columns[df2_cols == col].tolist()[0] for col in unmatched_cols],
        " Matching_col": ["No"] * len(unmatched_cols)
    })

    comparison_df = pd.concat([comparison_df, additional_rows], ignore_index=True)

    # Sort the DataFrame alphabetically by the column names of the first DataFrame
    comparison_df.sort_values(by=[df1_name, df2_name], inplace=True, ignore_index=True)

    print(comparison_df)

    #return comparison_df


# -------------------- ANALYSIS: Function to GET UNIQUE VALUES ----------------- #

def unique_values(data, dataset_name='Dataset'):
    """
    Analyze unique values
    =====================

    Analyze and print a summary of unique values for each column in the DataFrame or unique elements in a list.
    Handles columns with lists as values by converting them to strings.
    For columns or lists with a large number of unique values, only the count of unique values is returned.
    For columns or lists with fewer unique values, the unique values and their counts are listed in a readable format.

    Args:
    -----
    data (DataFrame or List): The pandas DataFrame or list to analyze.
    dataset_name (str): The name of the dataset being analyzed.
    """
    try:
        # Convert list to DataFrame
        if isinstance(data, list):
            data = pd.DataFrame(data, columns=["List_Values"])

        summaries = []
        for column in data.columns:
            # Convert lists to strings if necessary
            if data[column].apply(lambda x: isinstance(x, list)).any():
                data[column] = data[column].astype(str)

            unique_count = data[column].nunique()

            # If the number of unique values is too large, only return the count
            if unique_count > 15:  # adjust this threshold as needed
                summary = f"'{column}' has {unique_count} unique values"
            else:
                value_counts = data[column].value_counts().sort_values(ascending=False)
                value_summary = ', '.join([f"{val} ({count})" for val, count in value_counts.items()])
                summary = f"'{column}' has {unique_count} unique values: {value_summary}"

            summaries.append(summary)

        # Print the results
        print(f"\nUnique values and counts of {dataset_name}:")
        print('\n')
        print('\n\n'.join(summaries))
    except Exception as e:
        logging.error(f"Error in analyze_unique_values: {e}")


# -------------------- ANALYSIS: Function to find datetypes ----------------- #

def check_date_columns(df):
    """
    Check date information
    =======================

    Find columns in the DataFrame that potentially contain date information.

    This function checks for columns that are explicitly of datetime dtype,
    as well as columns with strings that may represent dates in various common formats.
    It uses dateutil for fuzzy parsing of date strings and regular expressions
    to match specific date patterns.

    Args:
    -----
    df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
    --------
    list: A list of column names that contain date information.

    Example usage:
    --------------
    date_columns = find_date_columns(your_dataframe)
    """
    def is_date(string, fuzzy=False):
        """
        Check if the string can be interpreted as a date.
        Uses dateutil.parser.parse to parse the string.

        Args:
        string (str): String to check for date.
        fuzzy (bool): If True, ignore unknown tokens in string.

        Returns:
        bool: True if string can be parsed as a date, False otherwise.
        """
        try: 
            parse(string, fuzzy=fuzzy)
            return True
        except ValueError:
            return False

    def date_format_match(string):
        """
        Check if the string matches common date formats using regular expressions.

        Args:
        string (str): String to check for date format.

        Returns:
        bool: True if string matches a common date format, False otherwise.
        """
        date_patterns = [
            r'\d{2}/\d{2}/\d{4}', r'\d{2}-\d{2}-\d{4}', r'\d{2}\.\d{2}\.\d{4}',
            r'\d{4}/\d{2}/\d{2}', r'\d{4}-\d{2}-\d{2}', r'\d{4}\.\d{2}\.\d{2}'
        ]
        for pattern in date_patterns:
            if re.match(pattern, string):
                return True
        return False

    try:
        date_columns = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_columns.append(col)
            elif df[col].apply(lambda x: is_date(str(x), fuzzy=True) and date_format_match(str(x))).any():
                date_columns.append(col)

        if not date_columns:
            logging.info("No date columns were found in the DataFrame.")
        else:
            print(f"Date columns in the DataFrame: {date_columns}")

        return date_columns

    except Exception as e:
        logging.error(f"Error in check_date_columns: {e}")
        return []


# -------------------- ANALYSIS: Function to GET UNIQUE VALUES ----------------- #

def check_unique_values(data, dataset_name='Dataset'):
    """
    Check unique values
    ====================

    Analyze and print a summary of unique values for each column in the DataFrame as a DataFrame.
    The summary includes the first unique values up to a maximum of 45 characters, the total count of unique values,
    and formatted Max & Min values.

    Args:
    ----
    data (pandas.DataFrame): The pandas DataFrame to analyze.
    dataset_name (str): The name of the dataset being analyzed. Defaults to 'Dataset'.

    Returns:
    --------
    None: This function prints the unique values summary directly.

    Example:
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1, 2, 2, 3], 'B': ['a', 'b', 'b', 'a']})
    >>> check_unique_values(data)
    """
    try:
        unique_summary = []

        for column in data.columns:
            unique_vals = data[column].dropna().unique()
            
            # Initialize max_val and min_val
            max_val = min_val = None

            # Sorting and data type checks
            if pd.api.types.is_numeric_dtype(data[column]):
                sorted_vals = sorted(unique_vals)
                max_val = f"{data[column].max():.2f}" if isinstance(data[column].max(), float) else data[column].max()
                min_val = f"{data[column].min():.2f}" if isinstance(data[column].min(), float) else data[column].min()
            elif pd.api.types.is_datetime64_any_dtype(data[column]):
                sorted_vals = sorted(unique_vals)
                max_val = data[column].max().strftime('%Y-%m-%d')
                min_val = data[column].min().strftime('%Y-%m-%d')
            elif pd.api.types.is_categorical_dtype(data[column]):
                try:
                    numeric_vals = pd.to_numeric(data[column].cat.categories, errors='raise')
                    sorted_vals = sorted(numeric_vals)
                    max_val = max(numeric_vals)
                    min_val = min(numeric_vals)
                except ValueError:
                    sorted_vals = sorted(data[column].cat.categories)
            else:
                sorted_vals = sorted(unique_vals)

            # Formatting first unique values to display
            formatted_values = []
            for val in sorted_vals:
                if pd.api.types.is_numeric_dtype(data[column]) or (pd.api.types.is_categorical_dtype(data[column]) and isinstance(val, (int, float))):
                    formatted_values.append(f"{val:.2f}" if isinstance(val, float) and val % 1 != 0 else val)
                elif pd.api.types.is_datetime64_any_dtype(data[column]):
                    formatted_values.append(val.strftime('%Y-%m-%d'))
                else:
                    formatted_values.append(val)

                # Join and truncate to max 45 characters
                unique_values = ', '.join(map(str, formatted_values))
                if len(unique_values) > 45:
                    unique_values = unique_values[:42] + "..."
                    break

            unique_count = len(unique_vals)
            dtype = data[column].dtype

            if max_val is None or min_val is None: 
                max_val = max_val if max_val is not None else '-'
                min_val = min_val if min_val is not None else '-'

            unique_summary.append([column, dtype, unique_values, unique_count, min_val, max_val])

        unique_df = pd.DataFrame(unique_summary, columns=['Column', 'Type', 'First Uniques', 'Count', 'Min', 'Max'])

        # Print the DataFrame
        print(f"\nUnique values summary for {dataset_name}:")
        print('\n')
        print(unique_df)
    except Exception as e:
        logging.error(f"Error in check_unique_values: {e}")


# -------------------- ANALYSIS: Function to check duplicates ----------------- #

def check_duplicates(df, dataset_name='Dataset'):
    """
    Check for duplicates
    ====================

    Check for duplicate rows in the DataFrame and print the results with total row count.

    Args:
    -----
    df (DataFrame): The pandas DataFrame to analyze.
    dataset_name (str): The name of the dataset being analyzed.
    """
    try:
        total_rows = len(df)
        duplicate_rows = df.duplicated().sum()
        message = f"No duplicate rows found out of total {total_rows} rows." if duplicate_rows == 0 else f"Out of {total_rows} rows, {duplicate_rows} are duplicates."
        print(f"Duplicates in {dataset_name}:\n{message}\n")
    except Exception as e:
        logging.error(f"Error in check_duplicates for {dataset_name}: {e}")
    

# ------------------- ANALYSIS: Function to check missing values ----------------- #

def check_missing_values(df, dataset_name='Dataset'):
    """
    Check for missing values
    ========================

    Check for missing values in the DataFrame and print the results.

    Args:
    ----
    df (DataFrame): The pandas DataFrame to analyze.
    dataset_name (str): The name of the dataset being analyzed.
    """
    try:
        missing_values = df.isnull().sum()
        message = "No missing values found." if missing_values.sum() == 0 else f"Missing values in the following columns:\n- {missing_values[missing_values > 0]}"
        print(f"Missing values in {dataset_name}:\n{message}\n")
    except Exception as e:
        logging.error(f"Error in check_missing_values for {dataset_name}: {e}")


# ---------------------- PREPROCESSING: Split Dataframe into Train and Test Sets ---------------------- #

def split_data(df, target, test_size=0.2, random_state=None, stratify=True):
    """
    Split dataframe
    ================

    Splits the dataframe into training and testing sets.

    Args:
    -----
    df (pd.DataFrame): The dataframe to split.
    target (str): The name of the target column for stratification.
    test_size (float or int, optional): If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples. Default is 0.2.
    random_state (int, optional): Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls. Default is None.
    stratify (bool, optional): If True, data is split in a stratified fashion, using the target. Default is True.

    Returns:
    --------
    pd.DataFrame, pd.DataFrame: The training and testing dataframes.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The provided dataset is not a DataFrame.")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    if not isinstance(test_size, (float, int)):
        raise ValueError("Test size must be float or int.")

    if isinstance(test_size, float) and not (0 < test_size < 1):
        raise ValueError("Test size must be between 0 and 1 when specified as a float.")

    if stratify and df[target].nunique() == 1:
        logging.warning("Stratification is not possible with a single unique target value. Proceeding without stratification.")
        stratify = False

    stratify_param = df[target] if stratify else None

    try:
        df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_param)
        logging.info(f"Data successfully split into training and testing sets. Training set size: {len(df_train)}, Testing set size: {len(df_test)}")
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise

    return df_train, df_test

# ---------------------- PREPROCESSING: Sorting columns ----------------- #

def sort_columns(
    datasets: pd.DataFrame, 
    capitalized: bool = True,
    case: Union[None, str] = None
) -> pd.DataFrame:
    """
    Sorts and Modifies DataFrame Columns
    ====================================
    
    This function sorts the columns of pandas DataFrames in-place within the provided dictionary in ascending order,
    with optional modifications to the case of column names. Additionally, it creates a summary DataFrame containing
    the dataset name, column number, datatype, and column name for each column.

    Parameters:
    -----------
    - datasets (dict of pd.DataFrame): Dictionary of DataFrames to sort column names. The key is the dataset name.
    - capitalized (bool, default=True): If True, capitalize the first letter of each column name before sorting.
    - case (str, optional): Specifies the case modification for column names. Accepts "lower", "upper", or None for no modification.

    Returns:
    --------
    - pd.DataFrame: A summary DataFrame with dataset name, column number, datatype, and column name.

    Raises:
    -------
    - ValueError: If `case` is not one of "lower", "upper", or None.
    - KeyError: If any provided DataFrame does not contain columns specified for modification.
    - Exception: If an error occurs during the sorting of DataFrame columns.

    Note:
    -----
    - The function modifies the column names according to the specified parameters before sorting.
    - It prints the sorted columns for each dataset and logs the shape of each DataFrame.

    Usage:
    ------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({'b_col': [1, 2, 3], 'a_col': [4, 5, 6], 'c_col': [7, 8, 9]})
    >>> df2 = pd.DataFrame({'d_col': [1, 2], 'a_col': [3, 4], 'c_col': [5, 6]})
    >>> datasets = {'dataset1': df1, 'dataset2': df2}

    >>> sorted_datasets = sort_columns(datasets, capitalized=True, case="upper")

    """
    # Validate case parameter
    if case not in {None, "lower", "upper"}:
        raise ValueError("Parameter 'case' must be one of 'lower', 'upper', or None.")

    sorted_datasets = {}

    try:
        for name, dataframe in datasets.items():
            # Validate DataFrame columns
            if not isinstance(dataframe, pd.DataFrame):
                raise ValueError(f"The dataset '{name}' is not a valid pandas DataFrame.")

            # Modify column names based on the specified parameters
            if capitalized:
                new_columns = {col: col[0].upper() + col[1:] for col in dataframe.columns}
                dataframe.rename(columns=new_columns, inplace=True)

            if case == "lower":
                dataframe.columns = [col.lower() for col in dataframe.columns]
            elif case == "upper":
                dataframe.columns = [col.upper() for col in dataframe.columns]

            # Sort columns after updating their names
            sorted_columns = sorted(dataframe.columns)
            dataframe = dataframe.reindex(columns=sorted_columns)
            sorted_datasets[name] = dataframe

            # Display the count of columns and rows
            logging.info(f"Shape of {name}: {dataframe.shape}")

            print(f"\n{name} - Sorted Features (A to Z):")
            for column in sorted_columns:
                print(column)
            print("\n" + "-"*45)

    except KeyError as e:
        logging.error(f"KeyError: {e}")
        raise
    except AttributeError as e:
        logging.error(f"AttributeError: {e}")
        raise
    except ValueError as e:
        logging.error(f"ValueError: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

    return sorted_datasets

# --------------------------- PREPROCESSING CLASS: Data Type Conversion -------------------- #

class DataPreprocessor:
    """
    Convert dtypes
    ===============

    A class for converting data types within a pandas DataFrame, optimizing memory usage,
    and applying both manual and automatic type conversions based on the characteristics
    of the data.
    
    Attributes:
    -----------
        dataset_name (str): Name of the dataset for identification and logging.
        manual_conversions (dict): A mapping of column names to their desired data types for manual conversion.
        auto_categorical_threshold (float): The threshold proportion of unique values to the total number of values,
                                            below which an object type column is automatically converted to 'category'.
        max_unique_values_for_category (int): The maximum number of unique values that an object type column can have to be 
                                              automatically converted to 'category'.
        exclude_from_auto_conversion (list): A list of column names to exclude from automatic conversion.
        
    Methods:
    --------
        convert_data_types(df): Applies type conversions to the given DataFrame and returns the modified DataFrame along with a summary.
    """
    
    def __init__(self, dataset_name='Dataset', manual_conversions=None, auto_categorical_threshold=0.5,
                 max_unique_values_for_category=1000, exclude_from_auto_conversion=None):
        """
        Initializes the DataPreprocessor with given configurations.
        """
        self.dataset_name = dataset_name
        self.manual_conversions = manual_conversions or {}
        self.auto_categorical_threshold = auto_categorical_threshold
        self.max_unique_values_for_category = max_unique_values_for_category
        self.exclude_from_auto_conversion = exclude_from_auto_conversion or []
    
    def convert_data_types(self, df):
        """
        Converts the data types in the DataFrame based on the instance configuration and optimizations.
        
        Parameters:
            df (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: The DataFrame with converted data types.
            pd.DataFrame: A summary DataFrame comparing data types before and after conversion.
        """
        # Store initial data types for comparison later
        before_conversion = df.dtypes.to_dict()
        
        # Manually convert specified columns
        for col, dtype in self.manual_conversions.items():
            if col in df.columns:
                try:
                    # Special handling for integer conversion
                    if dtype == 'int':
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                    else:
                        df[col] = df[col].astype(dtype)
                    logging.info(f"Manually converted {col} to {dtype}")
                except ValueError as e:
                    logging.error(f"Could not convert {col} to {dtype}: {e}")

        # Automatic data type conversion and memory optimization
        for col in df.columns:
            if col in self.exclude_from_auto_conversion:
                continue
            # Apply automatic conversions
            self._auto_convert_column(df, col)

        # Round off float columns to two decimal places
        df = self._round_float_columns(df)

        # Prepare a summary DataFrame for comparison
        comparison_df = self._prepare_comparison_dataframe(before_conversion, df)
        self._print_comparison(comparison_df)

        # Calculate and print the count of features with datatype "object" after conversion
        object_dtype_count = sum(df.dtypes == 'object')
        print(f"Number of features remaining with datatype 'object': {object_dtype_count}")
        print("\n" + "-"*75)

        return df

    def _auto_convert_column(self, df, col):
        """
        Automatically converts columns to the 'category' data type based on their properties.
        Also, applies downcasting for optimization and converts string columns representing
        dates or numbers to appropriate types.
        """
        # Convert date strings to datetime
        if 'date' in col.lower() and df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            logging.info(f"Converted {col} to datetime")

        # Convert numeric strings to appropriate numeric types
        if df[col].dtype == object and df[col].str.isnumeric().all():
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logging.info(f"Converted {col} to numeric")

        # Downcast numeric columns to optimize memory usage
        if df[col].dtype.kind in 'iuf':
            df[col] = pd.to_numeric(df[col], downcast='integer' if df[col].dtype.kind == 'i' else 'float')

        # Automatically convert to category if conditions are met
        if self._should_auto_convert(df[col]):
            df[col] = df[col].astype('category')
            logging.info(f"Automatically converted {col} to category")

    def _round_float_columns(self, df):
        """
        Rounds all float columns in the DataFrame to two decimal places.
        """
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = df[col].round(2)
        return df

    def _prepare_comparison_dataframe(self, before_conversion, df):
        """
        Prepares a DataFrame comparing data types before and after the conversion process.
        """
        comparison_df = pd.DataFrame(list(before_conversion.items()), columns=['Column', 'Before'])
        comparison_df['After'] = comparison_df['Column'].apply(lambda col: df[col].dtype)
        return comparison_df

    def _should_auto_convert(self, series):
        """
        Determines whether a column should be automatically converted to 'category'.
        """
        unique_count = series.nunique()
        total_count = len(series)
        is_eligible = unique_count / total_count < self.auto_categorical_threshold
        is_eligible &= unique_count <= self.max_unique_values_for_category
        is_eligible &= series.dtype == object
        return is_eligible

    def _print_comparison(self, comparison_df):
        """
        Prints a comparison of data types before and after conversion.
        """
        print(f"\n{self.dataset_name} - Data Type Conversion:")
        print(comparison_df)
        print("\n" + "-" * 75)

        logging.info(f"Data type conversion successful.\n")

# ---------------------- PREPROCESSING: Rename columns ----------------- #

def rename_and_compare_columns(dataframe, dataset_name,rename_map, ascending=True):
    """
    Rename columns of a pandas DataFrame according to a provided mapping 
    in a case-insensitive manner, sorts the columns, and prints comparison results 
    along with sorted column information.

    Args:
    dataframe (pd.DataFrame): DataFrame whose columns need to be renamed.
    rename_map (dict): Dictionary mapping from old column names (case-insensitive) to new column names.
    ascending (bool): If True, sort column names in ascending order; otherwise, sort in descending order.

    Returns:
    pd.DataFrame: DataFrame with renamed and sorted columns.
    """
    try:
        # Case-insensitive renaming
        current_columns = {col.lower(): col for col in dataframe.columns}
        new_rename_map = {}
        comparison_data = []

        for old_col, new_col in rename_map.items():
            actual_old_col = current_columns.get(old_col.lower())
            if actual_old_col:
                new_rename_map[actual_old_col] = new_col
                comparison_data.append({"Before": actual_old_col, "After": new_col})
            else:
                logging.warning(f"The column '{old_col}' is not found in the DataFrame (case-insensitive check). Skipping this column.")

        dataframe = dataframe.rename(columns=new_rename_map)
        logging.info(f"Columns renamed successfully in a case-insensitive manner.")

        # Sorting columns
        sorted_columns = sorted(dataframe.columns, reverse=not ascending)
        dataframe = dataframe[sorted_columns]

        # Print comparison results with dataset name
        print("\n" + "-"*70)
        print(f"\nColumn Rename Comparison for '{dataset_name}':")
        print(pd.DataFrame(comparison_data))

        # Print sorted columns
        column_summary = pd.DataFrame({
            "Data Type": [dataframe[col].dtype for col in sorted_columns],
            "Column Name": sorted_columns
        })
        print("\n")
        print(f"Sorted Columns in '{dataset_name}' ({'A to Z' if ascending else 'Z to A'}):")
        print(column_summary)

    except Exception as e:
        logging.error(f"Error in processing: {e}")
        raise

    return dataframe


# --------------------------- PREPROCESSING CLASS: DATA VALIDATION -------------------- #

class DataValidator:
    """
    Comprehensive Data Validator for DataFrames
    ============================================
    
    The `DataValidator` class is a sophisticated tool designed for comprehensive data validation 
    within a pandas DataFrame. It facilitates the enforcement of data integrity and conformity by 
    applying a set of predefined validation rules to the DataFrame's columns. This class is crucial 
    in data preprocessing workflows, where ensuring data quality and consistency directly influences 
    the accuracy and reliability of subsequent data analysis or machine learning models.

    Sections:
    ---------
    - Initialization
    - Column Exclusion Validation
    - Data Validation
    - Error Display
    - Utility Methods

    Initialization
    --------------
    
    Initializes the DataValidator with the specified parameters.

    Parameters:
    -----------
    - dataframe (pd.DataFrame): The DataFrame to validate.
    - rules (dict): Dictionary of validation rules for each column.
    - exclude_cols (list, optional): List of column names to exclude from validation. Defaults to None.
    - auto_validate (bool, optional): Whether to automatically validate the DataFrame upon initialization. Defaults to True.

    Column Exclusion Validation
    ---------------------------
    
    The `validate_exclude_cols` method processes and validates the list of columns to exclude from validation.

    Parameters:
    -----------
    - exclude_cols (list, optional): List of column names to exclude from validation.

    Returns:
    --------
    - list: Validated list of columns to exclude from validation.

    Raises:
    -------
    - ValueError: If the exclude_cols parameter is not a list or contains columns not found in the DataFrame.

    Data Validation
    ---------------
    
    The `validate` method validates the DataFrame columns based on the provided rules and aggregates errors.

    Parameters:
    -----------
    - show (bool, optional): Whether to log the number of different error types found. Defaults to True.

    Returns:
    --------
    - dict: A dictionary containing DataFrames with validation errors for each error type.

    Raises:
    -------
    - Exception: If an error occurs during the validation process.

    Error Display
    -------------
    
    The `display_errors` method displays validation errors using different methods based on the execution environment.

    Utility Methods
    ---------------
    
    - `_display_html`: Displays errors in HTML format in interactive environments.
    - `_display_text`: Displays errors in text format in non-interactive environments.
    - `_check_type`: Determines the likely data type of a column based on its content.
    - `_is_potentially_numeric`: Checks if a value is potentially numeric, considering date-like and time-like formats.
    - `_represents_alpha`: Checks if a value represents an alphabetical string.
    - `_represents_email`: Checks if a value represents an email.
    - `_represents_date`: Checks if a value represents a date.
    - `_represents_time`: Checks if a value represents a time format.
    - `_represents_ip`: Checks if a value represents a valid IP address.
    - `_check_special_chars`: Checks for unwanted special characters in a column.
    - `_contains_special_chars`: Helper to check for special characters in a string using vectorized operations.
    - `_check_accent_chars`: Checks for unwanted accent characters in a column of type 'alphabetical'.
    - `_contains_unwanted_accents`: Determines if a decomposed string contains any diacritical marks that are not explicitly allowed.
    - `_check_proper_case`: Ensures that names in a column follow a proper case according to natural naming conventions.
    - `_check_date_format`: Checks if dates in a column adhere to a specified format, ignoring special characters.
    - `_check_consistent_and_disproportional`: Checks if the numeric column's range is disproportionately wide.
    - `_represents_numeric`: Checks if a value represents a numeric value, including fractional representations.
    - `_calculate_dynamic_threshold`: Dynamically calculates a threshold ratio based on the statistics of the numeric values.
    - `_is_range_disproportional`: Checks if the range of values in a column is disproportionately wide.
    - `_check_email_format`: Validates the format of email addresses in a column.
    - `_check_whitespace_issues`: Checks for leading or trailing whitespaces in a DataFrame column using vectorized operations.
    - `_check_no_numerics_in_alphabetical`: Checks that no numeric values are present in columns classified as alphabetical.
    - `_check_no_alpha_in_numeric`: Ensures that no alphabetical characters are present in columns classified as numeric.
    - `_get_column_nature`: Determines the nature of the column based on the type of data it predominantly holds.
    - `_check_data_type`: Checks if data types in a column match the expected types in rules.
    - `_check_uniqueness`: Checks if values in a column are unique if required by rules.
    - `_check_range`: Checks if values in a numeric or date column fall within a specified range.
    - `_check_contains`: Checks if each value in the column contains a specified substring.

    Example Usage:
    --------------
    >>> import pandas as pd

    >>> df = pd.read_csv('data.csv')
    >>> rules = {
            'column1': {'data_type': str, 'unique': True, 'min': 0, 'max': 100},
            'column2': {'contains': '@'}
        }
    >>> validator = DataValidator(df, rules, exclude_cols=['id'])
    >>> validator.display_errors()

    Methods:
    --------
    """
    def __init__(self, dataframe, rules, exclude_cols=None, auto_validate=True):

        self.dataframe = dataframe
        self.rules = rules
        self.env_checker = EnvironmentChecker(show=False)
        self.exclude_cols = self.validate_exclude_cols(exclude_cols)

        if auto_validate:
            self.error_dfs = self.validate()
            self.display_errors()
        else:
            self.error_dfs = {}

    def validate_exclude_cols(self, exclude_cols):
        if exclude_cols is None:
            return []
        if not isinstance(exclude_cols, list):
            logging.info("exclude_cols should be a list of column names. Incorrect format provided.")
            return []
        
        # Check if all columns in exclude_cols are in the dataframe
        missing_cols = [col for col in exclude_cols if col not in self.dataframe.columns]
        if missing_cols:
            logging.info(f"Columns listed in exclude_cols not found in dataframe: {missing_cols}")
        
        return [col for col in exclude_cols if col in self.dataframe.columns]

    def __repr__(self):
        """
        Returns a string representation of the DataValidator object, showing the number of columns and rules applied.
        Returns:
            str: Description of the DataValidator instance.
        """
        return f"<DataValidator: {len(self.dataframe)} error types, {len(self.rules)} rules>"

# --------------------------------------------------------- DISPLAY CHECKS ------------------------------------------------------------- #

    def display_errors(self):
        """
        Display errors using different methods based on the execution environment.
        """
        if self.env_checker.is_interactive():
            # Use HTML display in interactive environments
            self._display_html()
        else:
            # Fallback to console display or logging in non-interactive environments
            self._display_text()

    def _display_html(self):
        # Existing HTML display logic
        fixed_width = '1000px'
        title_font_size = '16px'
        column_widths = {
            'Index': '50px',
            'Column': '150px',
            'Datatype': '100px',
            'Proposed': '100px',
            'Error': '400px',
            'Count': '50px'
        }
        headers_html = "<tr>" + "".join(f"<th style='width: {width}; text-align: left;'>{name}</th>" for name, width in column_widths.items()) + "</tr>"

        for error_type, df in self.error_dfs.items():
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Index'}, inplace=True)
            rows_html = "".join("<tr>" + "".join(f"<td style='width: {column_widths[col]}; text-align: left;'>{cell}</td>" for col, cell in row.items()) + "</tr>" for _, row in df.iterrows())
            table_html = f"<table style='width:{fixed_width}; table-layout: fixed; border-collapse: collapse;'>{headers_html}{rows_html}</table>"
            display(HTML(f"<h4 style='font-size:{title_font_size}; text-align: left;'>{error_type}</h4>{table_html}"))

    def _display_text(self):
        # Fallback display for text-based environments
        for error_type, df in self.error_dfs.items():
            print("\n" + f"{error_type}:")
            print(df.to_string(index=False))

# --------------------------------------------------------- VALIDATION CHECKS ------------------------------------------------------------- #

    def validate(self, show=True):
        """
        Validate the dataframe columns based on the provided rules. This function orchestrates the validation
        by iterating over each column, applying the checks defined in the rules, and aggregating any errors.

        Returns:
            dict: A dictionary containing error dataframes for each error type detected.
        """
        error_data = {}
        proposed_types = {}

        # Loop through each column in the validation rules
        for column, rule_details in self.rules.items():
            if column not in self.dataframe.columns:
                logging.info(f"Column '{column}' specified in rules is not found. Skipping validation_rule.")

        for column in self.dataframe.columns:
            if self.exclude_cols and column in self.exclude_cols:
                continue

            rules = self.rules.get(column, {})
            column_datatype = str(self.dataframe[column].dtype)
            column_errors = []

            # Attempt to determine the data type with error handling
            try:
                type_classification = self._check_type(self.dataframe[column])
                proposed_types[column] = type_classification  
            except Exception as e:
                logging.error(f"Error determining type for {column}: {e}")
                continue

            # Perform each check
            checks = [
                (self._check_data_type, [self.dataframe[column], rules]),
                (self._check_uniqueness, [self.dataframe[column], rules]),
                (self._check_special_chars, [self.dataframe[column], rules, proposed_types[column]]),
                (self._check_accent_chars, [self.dataframe[column], proposed_types[column]]),
                (self._check_consistent_and_disproportional, [self.dataframe[column], proposed_types[column]]),
                (self._check_proper_case, [self.dataframe[column], proposed_types[column]]),
                (self._check_email_format, [self.dataframe[column], proposed_types[column]]),
                (self._check_whitespace_issues, [self.dataframe[column]]),
                (self._check_no_numerics_in_alphabetical, [self.dataframe[column], proposed_types[column]]),
                (self._check_no_alpha_in_numeric, [self.dataframe[column], proposed_types[column]]),
                (self._check_date_format, [self.dataframe[column], rules, proposed_types[column]]),
                (self._check_contains, [self.dataframe[column], rules]), 
            ]

            # Range check, only if 'min' or 'max' is defined
            if 'min' in rules or 'max' in rules:
                checks.append((self._check_range, [self.dataframe[column], rules, proposed_types[column]]))

            for check_func, args in checks:
                try:
                    column_errors.extend(check_func(*args))
                except Exception as e:
                    logging.error(f"Error executing {check_func.__name__} for column {column}: {e}")

            # Aggregate error messages
            for error_type, error_values in column_errors:
                unique_error_values = list(set(error_values))
                count_errors = len(unique_error_values)
                error_data.setdefault(error_type, []).append({'Column': column, 'Datatype': column_datatype, 'Proposed': proposed_types[column], 'Error': unique_error_values, 'Count': count_errors})

        error_dfs = {error_type: pd.DataFrame(data) for error_type, data in error_data.items()}

        if show:
            # Logging the number of different error types found
            logging.info(f" {len(error_dfs)} different error types were detected in the dataframe")
        return error_dfs
    

    def _check_type(self, column):
        """
        Determine the likely data type of a column based on its content.

        Args:
        column (pd.Series): The column to classify.

        Returns:
        str: The identified type of the column.
        """
        numeric_count = 0
        alpha_count = 0
        email_count = 0
        date_count = 0
        time_count = 0
        ip_count = 0
        total_valid = 0

        for value in column.dropna():
            str_value = str(value).strip().rstrip('?')  # Standardize to string and strip trailing characters
            is_numeric = self._is_potentially_numeric(str_value)
            is_alpha = self._represents_alpha(str_value)
            is_email = self._represents_email(str_value)
            is_date = self._represents_date(str_value)
            is_time = self._represents_time(str_value)
            is_ip = self._represents_ip(str_value)

            numeric_count += is_numeric
            alpha_count += is_alpha and not is_numeric  # Only count as alpha if not numeric
            email_count += is_email
            date_count += is_date
            time_count += is_time
            ip_count += is_ip

            total_valid += 1

        # Debugging outputs
        #print(f"\nNumeric: {numeric_count}, Alpha: {alpha_count}, Email: {email_count}, Date: {date_count}, Time: {time_count}, IP: {ip_count}, Total: {total_valid}")

        if numeric_count > total_valid / 2:
            if date_count >= numeric_count * 0.5:  # If half of the numeric values are dates
                return 'date'
            elif time_count >= numeric_count * 0.5:  # If half of the numeric values are times
                return 'time'
            elif ip_count >= numeric_count * 0.5:  # More than half are IP addresses
                return 'ip_address'
            else:
                return 'numeric'
        elif alpha_count > total_valid / 2:
            if email_count > alpha_count * 0.5:
                return 'email'
            else:
                return 'alphabetical'
        else:
            return 'mixed'


    def _is_potentially_numeric(self, value):
        """Attempt to interpret a string as numeric, considering date-like and time-like formats."""
        # Regular expression to detect typical IP address patterns
        ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}(/\d{1,2})?$')

        if ip_pattern.match(value):
            # This matches an IP address pattern, so we return False for numeric
            return True
        try:
            # Try to convert directly to float
            float(value)
            return True
        except ValueError:
            # Handle fractions and date-like patterns
            try:
                # Check if it's a fraction (e.g., '1/2')
                Fraction(value)
                return True
            except ValueError:
                # Regex to check for date and time patterns
                if re.match(r'^\d{1,4}([-./])\d{1,2}\1\d{1,4}$', value):  # Date formats
                    return True
                if re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', value):  # Time formats
                    return True
                # Regex to identify strings with more digits than alphabetic characters
                digits = sum(c.isdigit() for c in value)
                letters = sum(c.isalpha() for c in value)
                if digits > letters:
                    return True
                return False

    def _represents_alpha(self, value):
        """ Check if the value represents an alphabetical string. """
        return isinstance(value, str) and any(c.isalpha() for c in value)

    def _represents_email(self, value):
        """ Check if the value represents an email. """
        if isinstance(value, str):
            parts = value.split('@')
            return len(parts) == 2 and all(part for part in parts)
        return False

    def _represents_date(self, value):
        """ Check if the value represents a date. """
        # A more comprehensive regex that attempts to capture most common date formats
        date_pattern = re.compile(
            r"""
            \b                      # Start at a word boundary
            (                       # Start of group:
            (?:                   # Try to match (non-capturing):
                \d{1,4}             # Year (1 to 4 digits)
                [-/\.]              # Separator
                \d{1,2}             # Month or day (1 or 2 digits)
                [-/\.]              # Separator
                \d{1,4}             # Day or year (1 to 4 digits)
            )                     # End non-capturing group
            |                     # OR
            (?:                   # Another non-capturing group:
                \d{1,2}             # Month or day (1 or 2 digits)
                [-/\.]              # Separator
                \d{1,2}             # Day or month (1 or 2 digits)
                [-/\.]              # Separator
                \d{2,4}             # Year (2 to 4 digits)
            )                     # End group
            )                       # End of first main group
            \b                      # End at a word boundary
            """, re.VERBOSE)

        if isinstance(value, str) and date_pattern.search(value):
            try:
                # Attempt to parse the date to confirm it's valid
                parsed_date = date_parser.parse(value, fuzzy=True)
                # Check if the parsed date is within a reasonable range (e.g., 1900-2099)
                return 1900 <= parsed_date.year <= 2099
            except (ValueError, TypeError):
                return False
        return False

    def _represents_time(self, value):
        """ Check if the value represents a time format. """
        if isinstance(value, str):
            # Regular expression to match HH:MM or HH:MM:SS
            time_pattern = re.compile(r'^\d{1,2}:\d{2}(:\d{2})?$')
            return bool(time_pattern.match(value))
        return False

    def _represents_ip(self, value):
        """ Check if the value represents a valid IP address. """
        ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}(/\d{1,2})?$')
        if ip_pattern.match(value):
            parts = value.split('.')
            return all(0 <= int(part) <= 255 for part in parts if part.isdigit())
        return False


# --------------------------------------------------------- AUTO CHECKS ------------------------------------------------------------- #

    def _check_special_chars(self, column, rules, column_type):
        """
        Check for unwanted special characters in a column.

        Args:
            column (pd.Series): The column to be checked.
            rules (dict): Dictionary containing validation rules for special characters.
            column_type (str): Type of the column (e.g., 'email', 'date') which could modify the set of characters to check.

        Returns:
            list: A list of errors found in the column.
        """
        column_errors = []
        default_special_chars = ['.', '?', '!', ';', '*', '+', '/', '{', '(', '§', '$', '%', ':', '@', '&', '#', '"', "'"]
        special_chars = rules.get('special_chars', default_special_chars)

        # Adjust special characters based on column type
        if column_type == 'email':
            special_chars = [char for char in special_chars if char not in ['@', '.']]
        elif column_type == 'date':
            special_chars = [char for char in special_chars if char not in ['.']]
        elif column_type == 'ip_address':
            special_chars = [char for char in special_chars if char not in ['.']]
        elif column_type == 'time':
            special_chars = [char for char in special_chars if char not in [':']]
        elif column_type == 'numeric':
            # For numeric columns, retain only those special characters that are unusual in numeric values
            special_chars = [char for char in special_chars if char not in ['.', '-', '+', 'e', 'E']]

        # Convert column to string if not already, to safely use .str accessor
        if column.dtype != 'object' and column.dtype.name != 'string':
            column = column.astype(str)

        # Check for special characters if not skipped by rules
        if not rules.get('skip_special_chars', False):
            error_mask = self._contains_special_chars(column, special_chars)
            if error_mask.any():
                error_values = column[error_mask].tolist()
                column_errors.append(("Special character check failed", error_values))
        return column_errors


    def _contains_special_chars(self, column, special_chars):
        """
        Helper to check for special characters in a string using vectorized operations.

        Args:
            column (pd.Series): The column of data to check.
            special_chars (list): List of special characters to check for in the column.

        Returns:
            pd.Series: A boolean series indicating which elements contain special characters.
        """
        pattern = '|'.join([re.escape(char) for char in special_chars])  # Escape to handle special regex characters
        contains_specials = column.str.contains(pattern, regex=True, na=False)
        return contains_specials

# ----------------------------------------------------------------------------------------------------------------------------------------- #

    def _check_accent_chars(self, column, column_type):
        """
        Check for unwanted accent characters in a column of type 'alphabetical'.
        Only specific diacritical marks are considered acceptable.

        Args:
            column (pd.Series): The column to be checked.
            column_type (str): Expected to be 'alphabetical' for this operation.

        Returns:
            list: A list of error messages detailing unwanted accent characters.
        """
        column_errors = []
        if column_type == 'alphabetical':
            try:
                # Convert to string to ensure consistent processing
                if column.dtype != 'object' and column.dtype != 'string':
                    column = column.astype(str)

                # Normalize and decompose each string to check for specific diacritical marks
                normalized = column.str.normalize('NFD')

                # Check each decomposed string for unwanted accents
                mask = normalized.map(self._contains_unwanted_accents)
                if mask.any():
                    error_values = column[mask].tolist()
                    column_errors.append(("Accent character check failed", error_values))
            except Exception as e:
                logging.error(f"Error checking accent characters in column: {e}")
                column_errors.append(("Error checking accent characters", str(e)))

        return column_errors

    def _contains_unwanted_accents(self, decomposed_string):
        """
        Determine if a decomposed string contains any diacritical marks that are not explicitly allowed.

        Args:
            decomposed_string (str): The Unicode Normalized (NFD) string to check.

        Returns:
            bool: True if unwanted diacritical marks are found, False otherwise.
        """
        allowed_accents = {'ä', 'ü', 'ö', 'Ä', 'Ü', 'Ö'}
        if isinstance(decomposed_string, str):
            try:
                for char in decomposed_string:
                    # Check if the character is a combining diacritical mark
                    if unicodedata.category(char).startswith('M') and unicodedata.normalize('NFC', 'a' + char) not in allowed_accents:
                        return True
                return False
            except TypeError:
                logging.error("Non-string input received while checking for unwanted accents.")
                return False
        else:
            return False  # Ignore non-string types
        
# ----------------------------------------------------------------------------------------------------------------------------------------- #

    def _check_proper_case(self, column, column_type):
        """
        Ensure that names in a column follow a proper case (capitalized properly) according to the rules of natural names
        and also allow for specialized identifiers such as enum names or constants which are typically used in programming
        and configuration files.

        Args:
        column (pd.Series): The column to be checked.
        column_type (str): The expected type of data in the column, used here to verify it's 'alphabetical'.

        Returns:
        list: A list of error messages with the associated incorrect entries if the column entries do not conform
            to the expected capitalization patterns.
        """
        column_errors = []

        if column_type == 'alphabetical':
            try:
                # Convert column to string if it's not already
                if column.dtype != 'object' and column.dtype != 'string':
                    column = column.astype(str)

                # Handle NaN values by replacing them with an empty string to avoid issues with regex operations
                column = column.fillna('')

                # Regex to match names that are properly capitalized, including those with diacritical marks
                pattern = re.compile(
                    r'^([A-ZÀÁÂÄÆÃÅĀÇĆČÈÉÊËĒĖĘÎÏÍĪĮÌÔÖÒÓŒØŌÕÚÜÙÛŪÑŃ]'  # Start with an uppercase letter (with diacritics)
                    r'[a-zàáâäæãåāçćčèéêëēėęîïíīįìôöòóœøōõúüùûūñń\'_]*'  # Followed by any number of lowercase letters (with diacritics), apostrophes, or underscores
                    r'([ -]'  # Space or hyphen delimiters, allowing for parts of names to be split by spaces or hyphens
                    r'[A-ZÀÁÂÄÆÃÅĀÇĆČÈÉÊËĒĖĘÎÏÍĪĮÌÔÖÒÓŒØŌÕÚÜÙÛŪÑŃ]'  # Next part starts with an uppercase letter
                    r'[a-zàáâäæãåāçćčèéêëēėęîïíīįìôöòóœøōõúüùûūñń\'_]*)*)$'  # Followed by lowercase letters, apostrophes, or underscores
                    r'|'  # OR allow for terms that are valid enum identifiers like 'Research_Development'
                    r'^([A-Z][A-Za-z0-9_]+)$'  # Enum style identifiers
                )

                # Create a mask where each valid item returns True; otherwise False
                valid_case_mask = column.str.match(pattern, na=False)

                # Find entries where valid_case_mask is False (invalid cases)
                invalid_cases = ~valid_case_mask

                if invalid_cases.any():
                    error_values = column[invalid_cases].tolist()
                    column_errors.append(("Proper case check failed", error_values))
            except Exception as e:
                logging.error(f"Error in _check_proper_case for column type {column_type}: {e}")
                column_errors.append(("Error during proper case validation", str(e)))

        return column_errors

# ----------------------------------------------------------------------------------------------------------------------------------------- #

    def _check_date_format(self, column, rules, column_type):
        """
        Checks if dates in a column adhere to a specified format, ignoring special characters.
        """
        column_errors = []
        if column_type == 'date':  # Only proceed if the column is classified as date type
            expected_format = rules.get('date_format', "%d.%m.%Y")  # Use specified format or default

            # Regex pattern to remove special characters except date separators
            # Allow date separators like . - /
            pattern = re.compile(r"[^0-9.\-\/]")

            # Convert all column entries to string to prevent attribute errors
            column = column.astype(str)  # Convert the entire column to strings

            for value in column:
                cleaned_value = pattern.sub('', value.strip())  # Remove special characters and strip whitespace
                try:
                    # Attempt to parse the date. If it fails, it means the format is not as expected.
                    datetime.strptime(cleaned_value, expected_format)
                except ValueError:
                    # Add to errors if parsing fails
                    column_errors.append(value)

            if column_errors:
                return [("Date format error, expected " + expected_format, column_errors)]
        return []

# ----------------------------------------------------------------------------------------------------------------------------------------- #

    def _check_consistent_and_disproportional(self, column, column_type):
        """
        Check if the numeric column's range is disproportionately wide.
        """
        column_errors = []
        if column_type == 'numeric':
            is_disproportional, dynamic_threshold, error = self._is_range_disproportional(column)
            if is_disproportional:
                column_errors.append((f"Range disproportion found with threshold: {dynamic_threshold:.2f}", error))
        return column_errors

    def _represents_numeric(self, value):
        """ Enhanced to handle fractional representations. """
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            try:
                # Try to convert fraction (e.g., '1/2' or '½')
                float(Fraction(str(value)))
                return True
            except (ValueError, TypeError):
                return False

    def _calculate_dynamic_threshold(self, numeric_values):
        """
        Dynamically calculates a threshold ratio based on the statistics of the numeric values.

        Args:
        numeric_values (pd.Series): Numeric values from a column.

        Returns:
        float: Calculated dynamic threshold ratio.
        """
        mean = numeric_values.mean()
        std_dev = numeric_values.std()

        # Prevent dividing by zero or excessively low means
        mean = max(abs(mean), 1e-6)  # Set a minimum mean value to avoid division issues

        cv = std_dev / mean
        skewness = numeric_values.skew()
        base_threshold = 10

        # Constrain the CV to prevent excessively high or low values
        cv = min(max(cv, 0.01), 100)

        # Adjust base threshold by half the CV
        adjustment_factor = 1 + 0.5 * cv

        if abs(skewness) > 1:  # Consider skewness for highly skewed distributions
            adjustment_factor *= 1.5

        # Constrain the dynamic threshold itself to prevent exceedingly high/low values
        dynamic_threshold = base_threshold * adjustment_factor
        dynamic_threshold = min(max(dynamic_threshold, 1), 1000)

        return dynamic_threshold

    def _is_range_disproportional(self, column, default_threshold_ratio=10):
        try:
            numeric_values = pd.to_numeric(column, errors='coerce')
            numeric_values.dropna(inplace=True)

            if numeric_values.empty:
                logging.debug("No valid numeric data to analyze.")
                return False, None, []

            dynamic_threshold = self._calculate_dynamic_threshold(numeric_values)
            min_value, max_value = numeric_values.min(), numeric_values.max()
            range_value = max_value - min_value
            unique_count = numeric_values.nunique()
            ratio = range_value / unique_count
            is_disproportional = ratio > dynamic_threshold
            #return is_disproportional, dynamic_threshold, [f"Uniques: {unique_count}, Min: {min_value}, Max: {max_value}, Range: {range_value:.0f}, Ratio: {ratio:.0f}"]
            return is_disproportional, dynamic_threshold, [f"Uniques: {unique_count}, Min: {min_value}, Max: {max_value}, Ratio: {ratio:.0f}"]
        except Exception as e:
            logging.error("Failed to calculate disproportionality: " + str(e))
            return False, None, ["Error calculating disproportionality: " + str(e)]

# ----------------------------------------------------------------------------------------------------------------------------------------- #

    def _check_email_format(self, column, column_type):
        """
        Validate the format of email addresses in a column.
        """
        column_errors = []
        if column_type == 'email':
            # Comprehensive regex for email validation:
            email_pattern = re.compile(
                r'^[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+'              # Local part: Usernames with acceptable characters
                r'@[a-zA-Z0-9-]+'                                  # Domain name part before the last dot
                r'(?:\.[a-zA-Z0-9-]+)*'                            # Subdomain(s): Allow multiple subdomains
                r'\.[a-zA-Z]{2,}$'                                 # Top-level domain: at least two letters
            )

            # Convert to string in case of non-string types and check against the regex
            format_issues = column.astype(str).str.match(email_pattern) == False
            
            if format_issues.any():
                # Collect errors
                column_errors.append(("Email format check failed", column[format_issues].tolist()))
        return column_errors


    def _check_whitespace_issues(self, column):
        """
        Check for leading or trailing whitespaces in a DataFrame column using vectorized operations.

        Args:
        column (pd.Series): The column to be checked.

        Returns:
        list: A list of tuples with error messages and the values that have whitespace issues.
        """
        column_errors = []
        try:
            # Ensure column is in string format if it's categorized as alphabetical
            if column.dtype != 'object' and column.dtype != 'string':
                column = column.astype(str)

            # Vectorized check for leading or trailing whitespaces
            leading_spaces = column.str.startswith(' ')
            trailing_spaces = column.str.endswith(' ')

            # Combine masks to find any leading or trailing whitespace issues
            whitespace_issues = leading_spaces | trailing_spaces

            if whitespace_issues.any():
                # Get unique entries with whitespace issues for reporting
                error_values = column[whitespace_issues].unique().tolist()
                column_errors.append(("Whitespace issues found", error_values))
        except Exception as e:
            # Log the exception details
            logging.error(f"Error checking whitespace in column: {e}")
            column_errors.append(("Error during whitespace validation", str(e)))

        return column_errors


    def _check_no_numerics_in_alphabetical(self, column, column_type):
        """
        Check that no numeric values are present in columns classified as alphabetical,
        distinguishing between numerics embedded in strings and purely numeric values.

        Parameters:
        - column (pd.Series): The column to be checked.
        - column_type (str): Expected to be 'alphabetical' for this operation.

        Returns:
        - list: A list containing tuples of error messages and the list of values that failed the validation.
        """
        column_errors = []

        if column_type != 'alphabetical':
            #logging.error(f"Incorrect column type '{column_type}' provided. Expected 'alphabetical'.")
            return column_errors

        try:
            # Check for purely numeric entries
            if column.dtype == 'int' or column.dtype == 'float':
                column_errors.append(("Numeric values in alphabetical column", column.tolist()))
            else:
                # Convert to string to ensure consistent processing
                column_str = column.astype(str)
                # Check for numeric characters using a regex pattern
                numeric_issues = column_str.str.contains(r'\d', regex=True)
                
                if numeric_issues.any():
                    # Collect values that have numeric characters
                    error_values = column[numeric_issues].tolist()
                    column_errors.append(("Numeric chars found in alphabetical column", error_values))
        except Exception as e:
            logging.error(f"Error checking numeric characters in column: {e}")
            column_errors.append(("Error during numeric character validation", str(e)))

        return column_errors


    def _check_no_alpha_in_numeric(self, column, column_type):
        """
        Ensure that no alphabetical characters are present in columns classified as numeric.
        """
        column_errors = []
        if column_type == 'numeric':
            # List of special characters to be excluded
            special_chars = ['?', '!',';', '+', '/', '%','@', '&', '"']

            # Define a check for special characters
            def contains_special_chars(s):
                return any(char in s for char in special_chars)

            # Define a check to see if the value is numeric
            def is_not_numeric(s):
                try:
                    float(s)  # Attempt to convert to float
                    return False  # It's numeric, no problem here
                except ValueError:
                    return True  # It's not numeric

            # Apply checks:
            non_numeric_issues = column.apply(lambda x: isinstance(x, str) and not contains_special_chars(x)) 
            
            if non_numeric_issues.any():
                column_errors.append(("Non-numeric chars found in numeric column", column[non_numeric_issues].tolist()))
        return column_errors


    def _get_column_nature(self, column):
        """
        Determine the nature of the column based on the type of data it predominantly holds.
        """
        if column.dtype.kind in 'biufc':
            return 'numeric'
        elif column.dtype.name == 'category':
            numeric_count = sum(pd.to_numeric(column.cat.categories, errors='coerce').notna())
            non_numeric_count = sum(pd.to_numeric(column.cat.categories, errors='coerce').isna())
            if numeric_count / (numeric_count + non_numeric_count) > 0.5:
                return 'numeric'
            else:
                return 'non_numeric'
        return 'mixed'
    
# --------------------------------------------------------- USER RULES ------------------------------------------------------------- #

    def _check_data_type(self, column, rules):
        """ Check if data types in a column match the expected types in rules. """
        column_errors = []
        if 'data_type' in rules:
            expected_dtype = rules['data_type']
            if not isinstance(column.dtype, expected_dtype):
                error_message = f"Expected datatype {expected_dtype}, found {column.dtype}"
                column_errors.append((error_message, column.tolist()))
        return column_errors

    def _check_uniqueness(self, column, rules):
        """ Check if values in a column are unique if required by rules. """
        column_errors = []
        if 'unique' in rules and rules['unique']:
            if column.duplicated().any():
                column_errors.append(("Duplicate values found", column[column.duplicated()].tolist()))
        return column_errors


    def _check_range(self, column, rules, column_type):
        """
        Check if values in a numeric or date column fall within a specified range,
        but only if 'min' or 'max' are explicitly provided in the rules and only for 'numeric' or 'date' column types.
        """
        # Skip if range is not defined or column type is inappropriate
        if 'min' not in rules and 'max' not in rules:
            logging.info(f"No range constraints provided for column: {column.name}")
            return []
        if column_type not in ['numeric', 'date']:
            logging.info(f"Skipping range check for non-numeric/date column: {column.name}")
            return []

        column_errors = []

        try:
            if column_type == 'numeric':
                # Convert column to numeric, safely ignoring non-numeric characters
                numeric_column = pd.to_numeric(column, errors='coerce')
                min_val = rules.get('min', numeric_column.min())
                max_val = rules.get('max', numeric_column.max())
                range_issues = (numeric_column < min_val) | (numeric_column > max_val)
            elif column_type == 'date':
                # Convert column to datetime, handling date format cleanly
                date_column = pd.to_datetime(column, errors='coerce')
                min_val = pd.to_datetime(rules.get('min', date_column.min()))
                max_val = pd.to_datetime(rules.get('max', date_column.max()))
                range_issues = (date_column < min_val) | (date_column > max_val)

            # Collect any rows where the range check fails
            if range_issues.any():
                column_errors.append(("Range check failed", column[range_issues].tolist()))
        except Exception as e:
            logging.error(f"Error processing range check for column {column.name}: {e}")
            column_errors.append(("Error processing range check", str(e)))

        return column_errors

    def _check_contains(self, column, rules):
        """
        Check if each value in the column contains a specified substring.

        Parameters:
        - column (pd.Series): The column to be checked.
        - rules (dict): Dictionary containing the 'contains' key with the substring as value.

        Returns:
        - list: A list containing tuples of error messages and the list of values that failed the validation.
        """
        column_errors = []
        substring = rules.get('contains')

        # Skip if no substring is specified
        if substring is None:
            #logging.info("No substring specified for checking.")
            return column_errors

        try:
            # Convert column to string and check if it contains the specified substring
            # We use case=False to make the check case-insensitive
            contains_issue = ~column.astype(str).str.contains(substring, case=False, na=False, regex=False)

            # Collect values that do not contain the substring
            if contains_issue.any():
                error_values = column[contains_issue].tolist()
                column_errors.append((f"Missing substring '{substring}'", error_values))

        except Exception as e:
            logging.error(f"Error during substring check: {e}")
            column_errors.append(("Error during substring check", str(e)))
        return column_errors

# --------------------------- PREPROCESSING: Removing Duplicates -------------------- #

def remove_duplicates(df, dataset_name):
    """
    Remove duplicate rows from a DataFrame and print before and after information.

    Args:
    df (DataFrame): The DataFrame from which to remove duplicates.
    dataset_name (str): The name of the dataset (for logging and printing purposes).

    Returns:
    DataFrame: A DataFrame with duplicates removed.
    """
    try:
        original_shape = df.shape
        df_cleaned = df.drop_duplicates().reset_index(drop=True)  # Reset the index here
        new_shape = df_cleaned.shape

        # Calculate the number of duplicates removed
        num_duplicates_removed = original_shape[0] - new_shape[0]

        # Print and log the results
        print(f"\n{dataset_name} - Duplicate Removal:")
        print(f"  Shape before removal: {original_shape}")
        print(f"  Shape after removal: {new_shape}")
        print(f"  Number of duplicates removed: {num_duplicates_removed}")

        if num_duplicates_removed > 0:
            print("\n")
            logging.info(f"Removed {num_duplicates_removed} duplicates from {dataset_name}.")
        else:
            logging.info(f"No duplicates found in {dataset_name}.")

        return df_cleaned

    except Exception as e:
        logging.error(f"Error in removing duplicates from {dataset_name}: {e}")
        return df  # Return the original DataFrame in case of an error



# ----------------- PREPROCESSING: Handling Missing Values ----------------- #

class MissingValueHandler:
    """
    Robust Missing Value Handler for DataFrames
    ===========================================
    
    This class provides comprehensive functionality for handling missing values in pandas DataFrames. It ensures the prevention 
    of data leakage and allows for the exclusion of specified columns from imputation calculations. The class supports fitting 
    the imputation strategy to the data, applying the imputation, and combining both operations for convenience.

    Attributes:
    -----------
    - target_variable (str): The name of the target column to be excluded from feature calculations.
    - exclude_columns (set): A set of columns to exclude from imputation calculations.

    Methods:
    --------
    - __init__(self, target_variable=None, exclude_columns=None):
        Initializes the MissingValueHandler instance with options to exclude certain columns from feature calculations.
    
    - _select_numerical_features(self, df):
        Selects numerical features from the DataFrame, excluding specified columns.

    - fit(self, df):
        Fits the handler by calculating the median for numerical features, excluding specified columns.

    - transform(self, df):
        Applies the calculated medians to impute missing values in the DataFrame and adds missing indicators.

    - fit_transform(self, df):
        Combines fitting and transforming in a single method for convenience.

    Parameters:
    -----------
    - target_variable (str, optional): The name of the target column to be excluded from feature calculations.
    - exclude_columns (list or set, optional): A list or set of columns to exclude from imputation calculations.

    Returns:
    --------
    - None

    Raises:
    -------
    - ValueError: If a specified column for exclusion is not found in the DataFrame.
    - TypeError: If the input is not a pandas DataFrame.
    - KeyError: If a specified column for imputation is not found in the DataFrame.
    - Exception: If an unexpected error occurs during the operations.

    Note:
    -----
    - The class ensures that specified columns are excluded from the imputation process.
    - It adds missing value indicators for imputed columns to preserve information about the data's missingness.

    Usage:
    ------
    >>> df = pd.read_csv('data.csv')
    >>> handler = MissingValueHandler(target_variable='target', exclude_columns=['id'])
    >>> df_imputed = handler.fit_transform(df)
    >>> print(df_imputed.head())
    """

    def __init__(self, target_variable=None, exclude_columns=None):
        """
        Initialize the MissingValueHandler instance with options to exclude certain columns from feature calculations.

        Parameters:
        -----------
        target_variable : str, optional
            The target variable column name to be excluded from the feature calculations.
        exclude_columns : list or set, optional
            A list or set of columns to exclude from imputation calculations.
        """
        try:
            self.target_variable = target_variable
            self.exclude_columns = set(exclude_columns) if exclude_columns else set()
            if target_variable:
                self.exclude_columns.add(target_variable)
            self.stats = {}
            logging.info("MissingValueHandler initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing MissingValueHandler: {e}")
            raise

    def _select_numerical_features(self, df):
        """
        Selects numerical features from the DataFrame, excluding specified columns.

        Parameters:
        -----------
        df : DataFrame
            The input pandas DataFrame containing the data.

        Returns:
        --------
        List[str]
            A list of numerical feature column names excluding specified columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        numerical_features = df.select_dtypes(include=['number']).columns.tolist()
        return [f for f in numerical_features if f not in self.exclude_columns]

    def fit(self, df):
        """
        Fit the handler by calculating median statistics for the specified DataFrame.

        Parameters:
        -----------
        - df (pd.DataFrame): The input pandas DataFrame containing the data.

        Returns:
        --------
        - self: Returns the current instance of MissingValueHandler.

        Raises:
        -------
        - KeyError: If a specified column for imputation is not found in the DataFrame.
        - ValueError: If a specified column for exclusion is not found in the DataFrame.
        - Exception: If an unexpected error occurs during the operation.
        """
        try:
            features = self._select_numerical_features(df)
            if not features:
                logging.warning("No numerical features available for median calculation.")
            self.stats = {feature: df[feature].median() for feature in features}
            logging.info(f"Calculated medians for features: {self.stats}")
            return self
        except KeyError as e:
            logging.error(f"KeyError in fit method: {e}")
            raise
        except ValueError as e:
            logging.error(f"ValueError in fit method: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in fit method: {e}")
            raise

    def transform(self, df):
        """
        Impute missing values in a DataFrame using the previously calculated statistics, add missing indicators, and return the modified DataFrame.

        Parameters:
        -----------
        - df (pd.DataFrame): The input pandas DataFrame containing the data.

        Returns:
        --------
        - pd.DataFrame: The modified pandas DataFrame with imputed missing values and added missing indicators.

        Raises:
        -------
        - TypeError: If the input is not a pandas DataFrame.
        - KeyError: If a specified column for imputation is not found in the DataFrame.
        - ValueError: If a specified column for exclusion is not found in the DataFrame.
        - Exception: If an unexpected error occurs during the operation.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        try:
            df_copy = df.copy()
            missing_before = df_copy.isnull().sum()

            # Impute values and add missing indicators
            for feature, median_value in self.stats.items():
                if feature in df_copy:
                    if df_copy[feature].isna().sum() > 0:
                        missing_indicator_column = f"{feature}_missing_indicator"
                        df_copy[missing_indicator_column] = df_copy[feature].isna().astype(int)
                        df_copy[feature].fillna(median_value, inplace=True)
                        logging.info(f"Imputed missing values in '{feature}' with median: {median_value}")
                else:
                    logging.warning(f"Feature '{feature}' not found in the input DataFrame.")

            # Ensure consistent indexing before creating comparison DataFrame
            missing_after = df_copy.isnull().sum()
            features_common = missing_before.index.intersection(missing_after.index)

            # Create comparison DataFrame using only consistent features
            comparison_df = pd.DataFrame({
                'Feature': features_common,
                'Missing Before': missing_before.loc[features_common].values,
                'Missing After': missing_after.loc[features_common].values
            })
            print(comparison_df)
            print('\n')
            return df_copy

        except KeyError as e:
            logging.error(f"KeyError in transform method: {e}")
            raise
        except ValueError as e:
            logging.error(f"ValueError in transform method: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in transform method: {e}")
            raise

    def fit_transform(self, df):
        """
        Fit the handler and transform the DataFrame in one step.

        Parameters:
        -----------
        - df (pd.DataFrame): The input pandas DataFrame containing the data.

        Returns:
        --------
        - pd.DataFrame: The modified pandas DataFrame with imputed missing values and added missing indicators.

        Raises:
        -------
        - TypeError: If the input is not a pandas DataFrame.
        - KeyError: If a specified column for imputation is not found in the DataFrame.
        - ValueError: If a specified column for exclusion is not found in the DataFrame.
        - Exception: If an unexpected error occurs during the operation.
        """
        try:
            self.fit(df)
            return self.transform(df)
        except Exception as e:
            logging.error(f"Unexpected error in fit_transform method: {e}")
            raise

    

# ---------------------- PREPROCESSING: Encoding Categorical Variables ---------------------- #

class CategoricalEncoder:
    """
    Categorical Encoder for DataFrames
    ================================================
    
    This class provides robust functionality for encoding categorical variables in a DataFrame, offering detailed control 
    over encoding methods and parameters. It supports various encoding techniques, including one-hot encoding, label encoding, 
    and custom mappings, while efficiently handling columns with high cardinality. The class is designed to ensure that 
    encoded data maintains integrity and consistency, and it offers extensive logging and result reporting.

    Attributes:
    -----------
    - target_column (str): The name of the target column for potential target-based encoding.
    - one_hot_threshold (int): The threshold for unique values below which one-hot encoding is applied.
    - auto_categorical_threshold (float): The ratio of unique values to total entries above which a column is considered to have high cardinality.
    - max_unique_values_for_category (int): Maximum number of unique values for a variable to still be considered low cardinality.
    - manual_encoding (dict): A dictionary specifying columns and their custom encoding methods and potential mappings.
    - exclude_columns (list): List of column names to be excluded from encoding processes.

    Methods:
    --------
    - __init__(self, target_column=None, one_hot_threshold=5, auto_categorical_threshold=0.05, max_unique_values_for_category=10, manual_encoding=None, exclude_columns=None):
        Initializes the CategoricalEncoder with specified parameters for encoding control.

    - identify_categorical_columns(self, df):
        Identifies categorical columns in the DataFrame that are not in the exclude_columns list, including boolean columns.

    - encode_columns(self, df, categorical_columns):
        Encodes categorical columns in a DataFrame according to predefined rules and the characteristics of each column.

    - encode_dataframe(self, df, dataset_name):
        Encodes the DataFrame's categorical variables and logs the results.

    Parameters:
    -----------
    - target_column (str, optional): The name of the target column for potential target-based encoding.
    - one_hot_threshold (int, default=5): The threshold for unique values below which one-hot encoding is applied.
    - auto_categorical_threshold (float, default=0.05): The ratio of unique values to total entries above which a column is considered to have high cardinality.
    - max_unique_values_for_category (int, default=10): Maximum number of unique values for a variable to still be considered low cardinality.
    - manual_encoding (dict, optional): A dictionary specifying columns and their custom encoding methods and potential mappings.
    - exclude_columns (list, optional): List of column names to be excluded from encoding processes.

    Returns:
    --------
    - None

    Raises:
    -------
    - ValueError: If a specified column for exclusion or manual encoding is not found in the DataFrame.

    Note:
    -----
    - The class is designed to handle both low and high cardinality categorical variables efficiently.
    - Extensive logging and result reporting ensure that the encoding process is transparent and easily traceable.

    Usage:
    ------
    >>> from sklearn.model_selection import train_test_split
    >>> df = pd.read_csv('data.csv')
    >>> encoder = CategoricalEncoder(target_column='target', one_hot_threshold=10, max_unique_values_for_category=15, manual_encoding={'column_name': {'method': 'label', 'mapping': {}}}, exclude_columns=['id'])
    >>> df_encoded = encoder.encode_dataframe(df, 'Dataset Name')
    >>> print(df_encoded.head())
    """
    def __init__(self, target_column=None, one_hot_threshold=5, auto_categorical_threshold=0.05, max_unique_values_for_category=10, manual_encoding=None, exclude_columns=None):
        self.target_column = target_column
        self.one_hot_threshold = one_hot_threshold
        self.auto_categorical_threshold = auto_categorical_threshold
        self.max_unique_values_for_category = max_unique_values_for_category
        self.manual_encoding = manual_encoding if manual_encoding else {}
        self.exclude_columns = exclude_columns if exclude_columns else []

    def identify_categorical_columns(self, df):
        """
        Identifies categorical columns in the DataFrame that are not in the exclude_columns list, including boolean columns.

        Parameters:
        -----------
        - df (pd.DataFrame): The DataFrame to analyze.

        Returns:
        --------
        - list: A list of categorical column names including boolean types.

        Raises:
        -------
        - ValueError: If an exclude column is not found in the DataFrame.
        """
        if self.exclude_columns:
            for col in self.exclude_columns:
                if col not in df.columns:
                    logging.info(f"Exclude column '{col}' not found in DataFrame.")
        return df.select_dtypes(include=['object', 'category', 'bool']).columns.difference(self.exclude_columns).tolist()

    def encode_columns(self, df, categorical_columns):
        """
        Encodes categorical columns in a DataFrame according to predefined rules and the characteristics of each column.

        Parameters:
        -----------
        - df (pd.DataFrame): The DataFrame containing the data to be encoded.
        - categorical_columns (list): A list of column names in the DataFrame that are to be encoded.

        Processes:
        ----------
        - The method iterates over each categorical column specified.
        - It checks the number of unique values to decide on the encoding method:
          - If a column's unique values exceed the max_unique_values_for_category, it logs a message and skips further encoding processes for that column.
          - If under the one_hot_threshold, one-hot encoding is applied.
          - If a custom mapping is provided for label encoding, it applies the mapping.
          - Otherwise, it applies default label encoding.
        - Converts any boolean columns to integer type to standardize data types.

        Returns:
        --------
        - tuple: A modified DataFrame with encoded columns and a list of newly added columns after one-hot encoding.

        Raises:
        -------
        - KeyError: If a specified column for encoding is not found in the DataFrame.
        """
        added_columns = []

        for col in categorical_columns:
            if col not in df.columns:
                logging.info(f"Column '{col}' specified for encoding not found in DataFrame.")
                continue

            unique_vals_count = df[col].nunique()
            col_encoding_method = self.manual_encoding.get(col, {}).get('method', 'one-hot')
            col_mapping = self.manual_encoding.get(col, {}).get('mapping', None)

            # Decision based on max_unique_values_for_category
            if unique_vals_count > self.max_unique_values_for_category:
                logging.info(f"Column '{col}' has high cardinality: {unique_vals_count} unique values.")
                # Maybe apply label encoding or other methods suitable for high cardinality
                continue  # or other logic

            # Apply one-hot if under one_hot_threshold and considered low cardinality
            if col_encoding_method == 'one-hot' and unique_vals_count <= self.one_hot_threshold:
                initial_columns = set(df.columns)
                df = pd.get_dummies(df, columns=[col], drop_first=False, prefix=col, dummy_na=False)
                new_columns = set(df.columns) - initial_columns
                added_columns.extend(new_columns)
            elif col_encoding_method == 'label':
                if col_mapping:
                    original_dtype = df[col].dtype
                    df[col] = df[col].astype('object')
                    df[col] = df[col].map(col_mapping)
                    if original_dtype.name.startswith('category'):
                        df[col] = df[col].astype('category').cat.add_categories([min(col_mapping.values()) - 1])
                    df[col] = df[col].fillna(min(col_mapping.values()) - 1).astype(int)
                else:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])

        # Validate that no boolean types remain in the DataFrame
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
        return df, added_columns

    def encode_dataframe(self, df, dataset_name):
        """
        Encodes the DataFrame's categorical variables and logs the results.

        Parameters:
        -----------
        - df (pd.DataFrame): The DataFrame to encode.
        - dataset_name (str): The name of the dataset for logging and identification purposes.

        Returns:
        --------
        - pd.DataFrame: The encoded DataFrame.

        Raises:
        -------
        - ValueError: If the target column is not found in the DataFrame.
        """
        if self.target_column and self.target_column not in df.columns:
            logging.info(f"Target column '{self.target_column}' not found in DataFrame.")

        categorical_columns = self.identify_categorical_columns(df)
        df_encoded, added_columns = self.encode_columns(df, categorical_columns)

        results = {
            'Dataset Name': dataset_name,
            'Categorical Columns': categorical_columns,
            'Original Shape': df.shape,
            'New Shape': df_encoded.shape,
            'Newly Added Columns': added_columns,
            'Newly Added Column Count': len(added_columns)
        }

        logging.info(f"Encoding completed successfully in {dataset_name}.")
        print(f"\nEncoding Results for {dataset_name}:")
        for key, value in results.items():
            print(f"{key}: {value}")
        print("\n" + "-"*75)

        return df_encoded


# ---------------------- PREPROCESSING: UPDATE & RENAME ---------------------- #

class DataFramePreprocessor:
    """
    Edit and Update DataFrame
    =========================
    
    A class to preprocess a DataFrame by renaming columns, dropping specified columns, 
    or selecting specific columns.

    Sections:
    ---------
    - Initialization
    - Column Renaming
    - Column Dropping
    - Column Selecting
    - DataFrame Processing
    - Utility Methods

    Initialization
    --------------

    Initializes the DataFramePreprocessor with specified configurations.

    Parameters:
    -----------
    - rename_map (dict, optional): Mapping of old column names to new names.
    - columns_to_drop (list, optional): List of column names to drop.
    - columns_to_select (list, optional): List of column names to select.

    Column Renaming
    ---------------

    Renames columns in the DataFrame based on the provided mapping.

    Methods:
    --------
    - `rename_columns`: Renames columns based on a provided mapping.

    Column Dropping
    ---------------

    Drops specified columns from the DataFrame.

    Methods:
    --------
    - `drop_columns`: Drops specified columns from the DataFrame.

    Column Selecting
    ----------------

    Selects specified columns from the DataFrame.

    Methods:
    --------
    - `select_columns`: Selects specified columns from the DataFrame.

    DataFrame Processing
    --------------------

    Processes the DataFrame by renaming, dropping, and selecting columns.

    Methods:
    --------
    - `process_dataframe`: Processes the DataFrame by applying renaming, dropping, and selecting operations.

    Utility Methods
    ---------------

    Example Usage:
    --------------
    >>> preprocessor = DataFramePreprocessor(rename_map={'old_name': 'new_name'}, columns_to_drop=['drop_col'], columns_to_select=['select_col'])
    >>> df_processed = preprocessor.process_dataframe(df, dataset_name='Sample Dataset')

    Attributes:
    -----------
    - rename_map (dict, optional): Mapping of old column names to new names.
    - columns_to_drop (list, optional): List of column names to drop.
    - columns_to_select (list, optional): List of column names to select.
    """
    def __init__(self, rename_map=None, columns_to_drop=None, columns_to_select=None):
        self.rename_map = rename_map
        self.columns_to_drop = columns_to_drop
        self.columns_to_select = columns_to_select

    def rename_columns(self, dataframe):
        """Rename columns based on a provided mapping."""
        if self.rename_map:
            missing_keys = [key for key in self.rename_map if key not in dataframe.columns]
            if missing_keys:
                logging.warning(f"These columns to rename are not in DataFrame: {missing_keys}")
            else:
                dataframe.rename(columns=self.rename_map, inplace=True)
                logging.info("Columns renamed successfully.")
        return dataframe

    def drop_columns(self, dataframe):
        """Drop specified columns from the DataFrame."""
        if self.columns_to_drop:
            # Identify columns that exist in the dataframe and can be dropped
            existing_cols = [col for col in self.columns_to_drop if col in dataframe.columns]

            # Identify and log columns that were requested for dropping but do not exist
            missing_cols = list(set(self.columns_to_drop) - set(existing_cols))
            if missing_cols:
                logging.warning(f"Attempted to drop non-existent columns: {missing_cols}")

            # Perform the drop operation for existing columns
            if existing_cols:
                dataframe.drop(columns=existing_cols, axis=1, inplace=True, errors='ignore')
                logging.info(f"Columns dropped successfully: {existing_cols}")
            else:
                logging.info("No existing columns were dropped because none of the specified columns were found.")

        return dataframe

    def select_columns(self, dataframe):
        """Select specified columns from the DataFrame."""
        if self.columns_to_select:
            existing_cols = [col for col in self.columns_to_select if col in dataframe.columns]
            if len(existing_cols) != len(self.columns_to_select):
                missing_cols = list(set(self.columns_to_select) - set(existing_cols))
                #logging.warning(f"Attempted to select non-existent columns: {missing_cols}")
                return dataframe  # Skip selecting and return the original dataframe
            dataframe = dataframe[existing_cols]
            logging.info("Columns selected successfully.")
        return dataframe

    def process_dataframe(self, dataframe, dataset_name):
        """Process the DataFrame by renaming, dropping, and selecting columns."""
        #logging.info(f"Processing {dataset_name}...")
        original_shape = dataframe.shape
        
        dataframe = self.rename_columns(dataframe)
        dataframe = self.drop_columns(dataframe)
        dataframe = self.select_columns(dataframe)
        
        new_shape = dataframe.shape
        logging.info(f"{dataset_name} processed from shape {original_shape} to {new_shape}.\n")

        return dataframe


# -------------------------- PREPROCESSING: Binning & Fixing Errors ------------------- #

class DataBinner:
    """
    Apply Binning to Numerical Data
    ===============================
    
    Handles the binning of numerical data in a DataFrame based on predefined binning rules.

    Sections:
    ---------
    - Initialization
    - Binning Application
    - Utility Methods

    Initialization
    --------------

    Initializes the DataBinner with specific binning rules.

    Parameters:
    -----------
    - binning_rules (dict): Specifies the binning rules for each column. Each key corresponds to a column name,
                            and the value is another dictionary with the keys 'bins', 'labels', and an optional 'right'
                            that indicates whether the bins include the rightmost edge.

    Binning Application
    -------------------

    Applies binning rules to the specified DataFrame. Logs a summary of the binning process once all applicable columns have been processed.

    Methods:
    --------
    - `apply_binning`: Applies the binning rules to the specified DataFrame.

    Utility Methods
    ---------------

    Example Usage:
    --------------
    >>> binning_rules = {
    >>>     'age': {'bins': [0, 18, 35, 50, 100], 'labels': ['Child', 'Young Adult', 'Adult', 'Senior'], 'right': False},
    >>>     'income': {'bins': [0, 30000, 60000, 100000], 'labels': ['Low', 'Middle', 'High']}
    >>> }
    >>> binner = DataBinner(binning_rules)
    >>> df_binned = binner.apply_binning(df, dataset_name='Sample Dataset')

    Attributes:
    -----------
    - binning_rules (dict): Specifies the binning rules for each column.
    """
    def __init__(self, binning_rules):
        """
        Initializes the DataBinner with specific binning rules.
        
        Parameters:
        -----------
        - binning_rules (dict): A dictionary containing the binning specifications for each column.
                                This includes the edges of the bins and optionally labels for these bins.
        """
        self.binning_rules = binning_rules

    def apply_binning(self, data, dataset_name):
        """
        Apply the binning rules to the specified DataFrame. Logs a summary of the binning process once all applicable columns have been processed.
        
        Parameters:
        -----------
        - data (pd.DataFrame): The dataset on which to apply the binning.
        - dataset_name (str): The name of the dataset for logging purposes, helps in identifying the dataset in logs.
        
        Returns:
        --------
        - pd.DataFrame: The dataset with additional columns for the binned data. Each binned column is named by appending '_Group' to the original column name.
        """
    def __init__(self, binning_rules):
        """
        Initializes the DataBinner with specific binning rules.
        
        Parameters:
            binning_rules (dict): A dictionary containing the binning specifications for each column.
                                  This includes the edges of the bins and optionally labels for these bins.
        """
        self.binning_rules = binning_rules

    def apply_binning(self, data, dataset_name):
        """
        Apply the binning rules to the specified DataFrame. Logs a summary of the binning process
        once all applicable columns have been processed.
        
        Parameters:
            data (pd.DataFrame): The dataset on which to apply the binning.
            dataset_name (str): The name of the dataset for logging purposes, helps in identifying the dataset in logs.
        
        Returns:
            pd.DataFrame: The dataset with additional columns for the binned data. Each binned column is named 
                          by appending '_Group' to the original column name.
        """
        binned_columns = []
        for column, rules in self.binning_rules.items():
            if column in data.columns:
                # Convert non-numeric values to the first bin's lower edge, handle missing values as well
                default_bin_value = rules['bins'][0]
                data[column] = pd.to_numeric(data[column], errors='coerce').fillna(default_bin_value)

                try:
                    bins = rules['bins']
                    labels = rules['labels']
                    right = rules.get('right', False)  # Default is True if not specified
                    ordered = not any(labels[i] == labels[i + 1] for i in range(len(labels) - 1))  # Set ordered=False if any labels are duplicated
                    new_column_name = f"{column}_Group"
                    data[new_column_name] = pd.cut(data[column], bins=bins, labels=labels, right=right, include_lowest=True, ordered=ordered)
                    binned_columns.append(column)
                except Exception as e:
                    logging.error(f"Error binning {column} in {dataset_name}: {e}")
            else:
                logging.warning(f"Column {column} not found in {dataset_name}. Skipping...")

        if binned_columns:
            logging.info(f"Binning successfully applied to {dataset_name}. Columns: {binned_columns}")
        else:
            logging.info(f"No binning was applied in {dataset_name} as no applicable columns were found.")

        return data


# ---------------------- PREPROCESSING: Feature Engineering ---------------------- #

class FeatureEngineer:
    """
    Feature Engineering
    ===================
    
    A class designed for feature engineering primarily focusing on time-series and transaction data. 
    It includes handling of date features, custom group-based operations, and feature selection based on top importance.

    Sections:
    ---------
    - Initialization
    - Column Handling
    - Date Feature Extraction
    - Custom Date Features
    - Training Data Transformation
    - Test Data Transformation
    - Utility Methods

    Initialization
    --------------

    Initializes the FeatureEngineer with specified configurations.

    Parameters:
    -----------
    - groupby_col (str, optional): Column name to group data before applying certain transformations.
    - date_columns_for_date_features (list, optional): List of column names that contain date information for feature extraction.
    - top_k_features (int, optional): Number of top features to select, if feature selection is applied.

    Column Handling
    ---------------

    Utility methods for handling columns in the DataFrame.

    Methods:
    --------
    - `drop_columns_with_suffix`: Removes columns from the dataframe that end with a specified suffix.
    - `find_date_columns`: Identifies and returns date columns in a DataFrame.

    Date Feature Extraction
    -----------------------

    Methods for extracting components from date columns.

    Methods:
    --------
    - `extract_date_components`: Extracts and adds year, month, day, and weekday components as new columns in the DataFrame.

    Custom Date Features
    --------------------

    Methods for adding custom date-related features like time since previous record and days since a reference date.

    Methods:
    --------
    - `add_date_related_features`: Adds custom date-related features to the DataFrame.

    Training Data Transformation
    ----------------------------

    Applies feature engineering to training data including dropping unnecessary columns, extracting date components,
    and adding custom date-related features.

    Methods:
    --------
    - `fit_transform`: Transforms the training dataset.

    Test Data Transformation
    ------------------------

    Applies the same transformations to test data that were applied during the training phase, using stored statistics.

    Methods:
    --------
    - `transform`: Transforms the test dataset.

    Utility Methods
    ---------------

    Example Usage:
    --------------
    >>> engineer = FeatureEngineer(groupby_col='user_id', date_columns_for_date_features=['purchase_date'], top_k_features=20)
    >>> df_train_transformed, train_stats = engineer.fit_transform(df_train)
    >>> df_test_transformed = engineer.transform(df_test)

    Attributes:
    -----------
    - groupby_col (str): Column name to group data before applying certain transformations.
    - date_columns_for_date_features (list): List of column names that contain date information for feature extraction.
    - top_k_features (int): Number of top features to select, if feature selection is applied.
    - train_stats (dict): Dictionary to store training statistics and information.
    """
    def __init__(self, groupby_col=None, date_columns_for_date_features=None, top_k_features=50):
        """
        Initializes the FeatureEngineer with specified configurations.

        Parameters:
            groupby_col (str): See class attributes.
            date_columns_for_date_features (list): See class attributes.
            top_k_features (int): See class attributes.
        """
        self.groupby_col = groupby_col
        self.date_columns_for_date_features = date_columns_for_date_features
        self.top_k_features = top_k_features
        self.train_stats = {}

    @staticmethod
    def drop_columns_with_suffix(df, suffix):
        """
        Removes columns from the dataframe that end with a specified suffix.

        Parameters:
            df (pd.DataFrame): DataFrame from which to drop columns.
            suffix (str): Suffix to match for dropping columns.

        Returns:
            pd.DataFrame: Modified DataFrame with specified columns removed.
        """
        return df.drop([col for col in df.columns if col.endswith(suffix)], axis=1, errors='ignore')

    @staticmethod
    def find_date_columns(df):
        """
        Identifies and returns date columns in a DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame to search for date columns.

        Returns:
            list: List of column names that are identified as date types.
        """
        return [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

    @staticmethod
    def extract_date_components(df, date_columns):
        """
        Extracts and adds year, month, day, and weekday components as new columns in the DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame from which to extract date components.
            date_columns (list): List of column names to process for date extraction.

        Returns:
            tuple: Modified DataFrame with new date component columns, List of new column names added.
        """
        added_columns = []
        for date_col in date_columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            for part in ['year', 'month', 'day']:
                new_col = f"{date_col}_{part}"
                df[new_col] = getattr(df[date_col].dt, part)
                added_columns.append(new_col)
            new_col = f"{date_col}_is_weekday"
            df[new_col] = (df[date_col].dt.dayofweek < 5).astype(int)
            added_columns.append(new_col)
        return df, added_columns

    def add_date_related_features(self, df, date_columns, reference_date=datetime(2022, 5, 1)):
        """
        Adds custom date-related features like time since previous record and days since a reference date.

        Parameters:
            df (pd.DataFrame): DataFrame to enhance with date-related features.
            date_columns (list): Columns on which to base the enhancements.
            reference_date (datetime): Reference date to calculate historical deltas.

        Returns:
            tuple: Updated DataFrame with new features, List of new column names added.
        """
        new_columns = []

        for date_column in date_columns:
            if date_column not in df.columns:
                logging.info(f"Column '{date_column}' not found in DataFrame. Skipping date-related feature creation.")
                continue
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

            valid_dates_df = df.dropna(subset=[date_column])
            if self.groupby_col and self.groupby_col in df.columns:
                valid_dates_df.sort_values(by=[self.groupby_col, date_column], inplace=True)

                if valid_dates_df.groupby(self.groupby_col)[date_column].nunique().gt(1).any():
                    valid_dates_df[f'previous_{date_column}'] = valid_dates_df.groupby(self.groupby_col)[date_column].shift(1)
                    new_col_name = f'time_gap_to_previous_{date_column}'
                    valid_dates_df[new_col_name] = (valid_dates_df[date_column] - valid_dates_df[f'previous_{date_column}']).dt.days
                    df = df.merge(valid_dates_df[[new_col_name]], left_index=True, right_index=True, how='left')
                    new_columns.append(new_col_name)

            days_col = f'days_since_{date_column}'
            months_col = f'months_since_{date_column}'
            valid_dates_df[days_col] = (reference_date - valid_dates_df[date_column]).dt.days
            valid_dates_df[months_col] = np.round(valid_dates_df[days_col] / 30).astype(int)
            df = df.merge(valid_dates_df[[days_col, months_col]], left_index=True, right_index=True, how='left')
            new_columns.extend([days_col, months_col])
        return df, new_columns

    def fit_transform(self, df_train):
        """
        Applies feature engineering to training data including dropping unnecessary columns, extracting date components,
        and adding custom date-related features.

        Parameters:
            df_train (pd.DataFrame): The training dataset to process.

        Returns:
            tuple: The transformed DataFrame and a dictionary of transformation results and stats.
        """
        results = {
            'Added Date Features': [],
            'Added Time Gap Features': [],
            'Original Shape': df_train.shape,
            'New Shape': None,
            'All Added Columns': [],
            'Added Column Count': 0
        }
        df_train = self.drop_columns_with_suffix(df_train, 'missing_indicator')
        date_columns = self.find_date_columns(df_train)
        original_columns = set(df_train.columns)
        if self.date_columns_for_date_features:
            df_train, time_added_cols = self.add_date_related_features(df_train, self.date_columns_for_date_features)
            results['Added Time Gap Features'].extend(time_added_cols)
        if date_columns:
            df_train, date_added_cols = self.extract_date_components(df_train, date_columns)
            results['Added Date Features'].extend(date_added_cols)

        results['New Shape'] = df_train.shape

        new_columns = set(df_train.columns) - original_columns
        results['All Added Columns'] = list(new_columns)
        results['Added Column Count'] = len(new_columns)
        self.train_stats['results'] = results
        self.train_stats['new_columns'] = new_columns

        print(f"Train - Feature Engineering Results:")
        for key, value in results.items():
            print(f"{key}: {value}")
            if key == 'Added Time Gap Features':
                print()  # Adding a line break
        print("\n" + "-"*50)  # extra line break for separation

        return df_train, results

    def transform(self, df_test):
        """
        Applies the same transformations to test data that were applied during the training phase, using stored statistics.

        Parameters:
            df_test (pd.DataFrame): The test dataset to process.

        Returns:
            pd.DataFrame: The transformed test dataset.
        """
        results = {
            'Added Date Features': [],
            'Added Time Gap Features': [],
            'Original Shape': df_test.shape,
            'New Shape': None,
            'All Added Columns': [],
            'Added Column Count': 0
        }
        df_test = self.drop_columns_with_suffix(df_test, 'missing_indicator')
        date_columns = self.find_date_columns(df_test)
        original_columns = set(df_test.columns)
        if self.date_columns_for_date_features:
            df_test, time_added_cols = self.add_date_related_features(df_test, self.date_columns_for_date_features)
            results['Added Time Gap Features'].extend(time_added_cols)
        if date_columns:
            df_test, date_added_cols = self.extract_date_components(df_test, date_columns)
            results['Added Date Features'].extend(date_added_cols)

        results['New Shape'] = df_test.shape

        new_columns = set(df_test.columns) - original_columns
        results['All Added Columns'] = list(new_columns)
        results['Added Column Count'] = len(new_columns)

        new_columns = self.train_stats.get('new_columns', set())
        for col in new_columns:
            if col not in df_test.columns:
                df_test[col] = np.nan

        print(f"Test - Feature Engineering Results:")
        for key, value in results.items():
            print(f"{key}: {value}")
            if key == 'Added Time Gap Features':
                print()  # Adding a line break
        print("\n" + "-"*50)  # extra line break for separation

        return df_test


# ----------------- PREPROCESSING: Aggregate and Mode Imputation Enhancements (Feature Augmentation) ----------------- #

class FeatureAggregator:
    """
    Feature Aggregation
    ===================
    
    A class designed for feature aggregation, primarily focusing on grouping data and applying various aggregation
    functions. It supports handling binary columns, preparing aggregation functions, and transforming datasets with 
    the specified aggregations.

    Sections:
    ---------
    - Initialization
    - Aggregation Preparation
    - Model Fitting
    - Data Transformation
    - Utility Methods

    Initialization
    --------------

    Initializes the FeatureAggregator with specified configurations.

    Parameters:
    -----------
    - agg_cols_dict (dict): Dictionary mapping column names to lists of aggregation functions.
    - groupby_col (str): Column name to group the DataFrame by before aggregating.

    Aggregation Preparation
    -----------------------

    Prepares aggregation functions based on the provided DataFrame and checks for column existence.

    Methods:
    --------
    - `_prepare_aggregations`: Validates and prepares aggregation functions for all required columns.

    Model Fitting
    -------------

    Fits the aggregator on the training data to determine appropriate aggregation functions.

    Methods:
    --------
    - `fit`: Fits the aggregator to the training dataset.

    Data Transformation
    -------------------

    Transforms a DataFrame using the fitted aggregation functions.

    Methods:
    --------
    - `transform`: Transforms the dataset with specified aggregations.

    Utility Methods
    ---------------

    - `print_results`: Prints formatted results of the aggregation process.

    Example Usage:
    --------------
    >>> aggregator = FeatureAggregator(agg_cols_dict={'amount': ['sum', 'mean'], 'transactions': ['count']}, groupby_col='user_id')
    >>> aggregator.fit(df_train)
    >>> df_train_agg = aggregator.transform(df_train, dataset_name='Training Data')
    >>> df_test_agg = aggregator.transform(df_test, dataset_name='Test Data')

    Attributes:
    -----------
    - agg_cols_dict (dict): Dictionary mapping column names to lists of aggregation functions.
    - groupby_col (str): Column name to group the DataFrame by before aggregating.
    - binary_means_to_convert (list): Stores names of binary columns to convert mean aggregates to integers.
    - train_agg_funcs (dict): Stores aggregation functions validated and adjusted during fitting.
    - supported_funcs (list): List of supported aggregation functions.
    """
    def __init__(self, agg_cols_dict, groupby_col):
        """
        Initializes the FeatureAggregator.

        Parameters:
        agg_cols_dict (dict): Dictionary mapping column names to lists of aggregation functions.
        groupby_col (str): Column name to group the DataFrame by before aggregating.

        Attributes:
        binary_means_to_convert (list): Stores names of binary columns to convert mean aggregates to integers.
        train_agg_funcs (dict): Stores aggregation functions validated and adjusted during fitting.
        supported_funcs (list): List of supported aggregation functions.
        """
        self.agg_cols_dict = agg_cols_dict
        self.groupby_col = groupby_col
        self.binary_means_to_convert = []
        self.train_agg_funcs = None
        self.supported_funcs = ['mean', 'median', 'std', 'sum', 'count', 'conditional_mean']

    def _prepare_aggregations(self, df, dataset_name, check_unique_cols=None):
        """
        Prepares aggregation functions based on the provided DataFrame and checks for column existence.
        """
        agg_functions = {}
        # Identify binary columns not explicitly mentioned for aggregation by mean
        binary_cols = [col for col in df.columns if df[col].dropna().isin([0, 1]).all() and col not in self.agg_cols_dict]
        for col in binary_cols:
            self.agg_cols_dict[col] = ['mean']
            self.binary_means_to_convert.append(col)

        # Validate and prepare aggregation functions for all required columns
        for col, funcs in self.agg_cols_dict.items():
            if col not in df.columns:
                logging.warning(f"{dataset_name}: Column '{col}' not found in DataFrame, skipping.")
                continue
            if check_unique_cols and col in check_unique_cols and df[col].nunique() != 1:
                logging.warning(f"{dataset_name}: Column '{col}' does not have unique values for each group, skipping.")
                continue
            valid_funcs = [func for func in funcs if func in self.supported_funcs]
            if valid_funcs:
                agg_functions[col] = valid_funcs

        return agg_functions

    def fit(self, df_train, check_unique_cols=None):
        """
        Fits the aggregator on the training data to determine appropriate aggregation functions.
        """
        logging.info("Fitting aggregator to training data...")
        self.train_agg_funcs = self._prepare_aggregations(df_train, "Training Data", check_unique_cols)

    def print_results(self, results):
        """
        Prints formatted results of the aggregation process.
        """
        print("Aggregation Results:")
        for key, value in results.items():
            if isinstance(value, list):
                print(f"{key}:")
                for item in value:
                    print(f" - {item}")
            else:
                print(f"{key}: {value}")
        print("-" * 50)

    def transform(self, df, dataset_name):
        """
        Transforms a DataFrame using the fitted aggregation functions.
        """
        if not self.train_agg_funcs:
            raise ValueError("The aggregator has not been fitted yet.")

        current_agg_funcs = {col: funcs for col, funcs in self.train_agg_funcs.items() if col in df.columns}
        try:
            grouped = df.groupby(self.groupby_col)
            agg_df = grouped.agg(current_agg_funcs)
            agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
            agg_df.reset_index(inplace=True)
            agg_df = agg_df.round(2)

            # Convert means for binary columns back to integers
            for col in self.binary_means_to_convert:
                col_name = f"{col}_mean"
                if col_name in agg_df.columns:
                    agg_df[col_name] = agg_df[col_name].astype(int)

            logging.info(f"Custom aggregated features added successfully in {dataset_name}.\n")
        except Exception as e:
            logging.error(f"Error during aggregation of {dataset_name}: {str(e)}\n")
            raise

        results = {
            'Original Shape': df.shape,
            'New Shape': agg_df.shape,
            #'Aggregated Columns': list(current_agg_funcs.keys()),
            'Newly aggregated cols Count': len(current_agg_funcs)
        }
        self.print_results(results)
        return agg_df


# ---------------------- PREPROCESSING: Data Standardization ---------------------- #

class DataScaler:
    """
    Data Scaler for DataFrames
    ========================================
    
    This class provides robust functionality for scaling numerical columns in a DataFrame, offering detailed control 
    over the scaling methods and excluding specific columns from the scaling process. It supports various scaling techniques 
    including z-score, min-max, and robust scaling, ensuring that data is standardized appropriately for further analysis.

    Sections:
    ---------
    - Initialization
    - Column Selection
    - Scaling Methods
    - Fit and Transform
    - Utility Methods

    Initialization
    --------------
    
    Initializes the DataScaler with the specified parameters.

    Parameters:
    -----------
    - target (str, optional): Name of the target column to exclude from scaling. Defaults to None.
    - exclude_cols (list, optional): List of additional column names to exclude from scaling. Defaults to None.
    - method (str): Scaling method. Supported values are "z-score", "min-max", "robust". Defaults to "z-score".

    Column Selection
    ----------------
    
    The columns to be scaled are determined based on the specified parameters, excluding the target column and any
    additional columns specified in exclude_cols.

    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    --------
    - list: List of column names to be scaled.

    Raises:
    -------
    - ValueError: If the input DataFrame is not a pandas DataFrame.

    Scaling Methods
    ---------------
    
    Supports multiple scaling methods:
    - Z-Score
    - Min-Max
    - Robust

    Fit and Transform
    -----------------
    
    The `fit` method fits the scaler on the specified columns in the DataFrame. The `transform` method applies the fitted
    scaler to scale the DataFrame.

    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame used to fit or transform the scaler.

    Returns:
    --------
    - pd.DataFrame: Scaled DataFrame after applying the transform method.

    Raises:
    -------
    - ValueError: If the input DataFrame is not a pandas DataFrame or if an unknown scaling method is provided.
    - RuntimeError: If the scaler has not been fitted before calling the transform method.
    - Exception: If an error occurs during the fitting or transformation process.

    Utility Methods
    ---------------
    
    - `fit`: Fits the scaler based on the DataFrame's specified columns, excluding the target and exclude_cols.
    - `transform`: Applies the fitted scaler to the DataFrame, scaling specified columns.

    Example Usage:
    --------------
    >>> import pandas as pd
    >>> from sklearn.model_selection import train_test_split

    >>> df = pd.read_csv('data.csv')
    >>> scaler = DataScaler(target='target', exclude_cols=['id'], method='z-score')
    >>> scaler.fit(df)
    >>> df_scaled = scaler.transform(df)
    >>> print(df_scaled.head())

    Methods:
    --------
    """
    def __init__(self, target=None, exclude_cols=None, method="z-score"):
        """
        Initializes the DataScaler with a specific scaling method, target column, and columns to exclude from scaling.

        Parameters:
        - target (str, optional): Name of the target column to exclude from scaling. Defaults to None.
        - exclude_cols (list, optional): List of additional column names to exclude from scaling. Defaults to None.
        - method (str): Scaling method. Supported values are "z-score", "min-max", "robust". Defaults to "z-score".
        """
        self.target = target
        self.exclude_cols = exclude_cols or []
        self.method = method
        self.scaler = None
        self.columns_to_scale = []  # This will store the columns that we fit the scaler on

    def fit(self, df):
        """
        Fits the scaler based on the DataFrame's specified columns, excluding the target and exclude_cols.

        Parameters:
        - df (pd.DataFrame): The DataFrame used to fit the scaler.
        """
        # Validate input DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # Determine columns to scale
        self.columns_to_scale = [col for col in df.select_dtypes(include=np.number).columns 
                                if col not in self.exclude_cols + ([self.target] if self.target else [])]

        # Initialize the scaler based on the specified method
        if self.method == "z-score":
            self.scaler = StandardScaler()
        elif self.method == "min-max":
            self.scaler = MinMaxScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method '{self.method}'. Supported methods: z-score, min-max, robust.")
        
        # Fit the scaler on the specified columns
        try:
            self.scaler.fit(df[self.columns_to_scale])
            logging.info("Scaler fitted successfully.")
        except Exception as e:
            logging.error("Error occurred during fitting scaler: ", exc_info=True)
            raise e

    def transform(self, df):
        """
        Applies the fitted scaler to the DataFrame, scaling specified columns.

        Parameters:
        - df (pd.DataFrame): The DataFrame to scale.

        Returns:
        - pd.DataFrame: Scaled DataFrame.
        """
        if not self.scaler:
            raise RuntimeError("Scaler has not been fitted. Call 'fit' with training data first.")
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        df_scaled = df.copy()
        
        # Directly use columns determined during fitting to ensure consistency
        try:
            df_scaled[self.columns_to_scale] = self.scaler.transform(df[self.columns_to_scale])
            logging.info("Scaling applied successfully.")
        except Exception as e:
            logging.error("Error occurred during scaling: ", exc_info=True)
            raise e

        return df_scaled


# ---------------------- PREPROCESSING: Handling and Detecting Outliers ---------------------- #

class OutlierHandler:
    """
    Outlier Detection and Handling Class
    ====================================
    
    This class implements various methods for detecting and handling outliers in a DataFrame. It supports multiple 
    outlier detection methods including z-score, IQR, and Isolation Forest. Additionally, it provides methods for 
    flagging outliers and visualizing them.

    Sections:
    ---------
    - Initialization
    - Outlier Detection Methods
    - Data Preprocessing
    - Model Training
    - Outlier Flagging
    - Visualization
    - Utility Methods

    Initialization
    --------------
    
    Initializes the OutlierHandler with the specified parameters.

    Parameters:
    -----------
    - target_column (str): Name of the target column, to be excluded from outlier analysis.
    - exclude_cols (List[str], optional): Columns to exclude from the outlier analysis.
    - unique_value_threshold (int): Threshold for the number of unique values to consider a feature continuous.

    Outlier Detection Methods
    -------------------------
    
    Supports multiple outlier detection methods:
    - Z-Score
    - IQR (Interquartile Range)
    - Isolation Forest
    - DBSCAN

    Data Preprocessing
    ------------------
    
    The `identify_feature_types` method categorizes the features in the DataFrame based on the number of unique values.

    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame to analyze.
    - unique_value_threshold (int): Threshold for unique values to consider a feature continuous.

    Returns:
    --------
    - dict: Categorized lists of feature names.

    Model Training
    --------------
    
    The `fit` method fits the outlier detection model on the provided DataFrame using the specified method.

    Parameters:
    -----------
    - df (pd.DataFrame): The training DataFrame.
    - method (str): Outlier detection method ('zscore', 'iqr', 'isolation_forest').

    Raises:
    -------
    - ValueError: If an unsupported method is provided.

    Outlier Flagging
    ----------------
    
    The `apply_outlier_flags` method identifies outliers in a DataFrame using the fitted model or calculated statistics.

    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame to analyze for outliers.

    Returns:
    --------
    - pd.DataFrame: DataFrame with added outlier-related features ('outlier_count', 'Is_outlier', 'outlier_coefficient').

    Visualization
    -------------
    
    The `visualize_outliers` method visualizes outliers in the DataFrame's numeric columns based on the detection method.

    Parameters:
    -----------
    - df (pd.DataFrame): DataFrame containing the data.

    Raises:
    -------
    - ValueError: If detection method has not been set before visualization.

    Utility Methods
    ---------------
    
    - `detect_outliers`: Detects outliers in the DataFrame based on the fitted model or statistics.
    - `calculate_feature_stats`: Calculates necessary statistics for z-score or IQR methods.
    - `flag_outliers_z_score`: Flags outliers based on Z-Score.
    - `flag_outliers_iqr`: Flags outliers based on the Interquartile Range (IQR).
    - `flag_outliers_isolation_forest`: Flags outliers using the Isolation Forest method.
    - `flag_outliers_dbscan`: Flags outliers using the DBSCAN clustering method.

    Example Usage:
    --------------
    >>> handler = OutlierHandler(target_column='target', exclude_cols=['id'], unique_value_threshold=10)
    >>> handler.fit(df, method='zscore')
    >>> outliers_df = handler.apply_outlier_flags(df)
    >>> handler.visualize_outliers(df)

    Methods:
    --------
    """
    def __init__(self, target_column='', exclude_cols=None, unique_value_threshold=10):
        """
        Initializes the OutlierHandler with configurations for handling outliers.

        Parameters:
        - target_column (str): Name of the target column, to be excluded from outlier analysis.
        - exclude_cols (List[str]): Columns to exclude from the outlier analysis.
        - unique_value_threshold (int): Threshold for the number of unique values to consider a feature continuous.
        """
        self.target_column = target_column
        self.exclude_cols = exclude_cols if exclude_cols else []
        self.unique_value_threshold = unique_value_threshold
        self.detection_method = None
        self.model = None  # For model-based outlier detection methods
        self.stats = {}  # For storing stats for methods like zscore or IQR

    def fit(self, df, method):
        """
        Fit the outlier detection based on the provided method using training data.

        Parameters:
        - df (pd.DataFrame): Training data.
        - method (str): Outlier detection method ('zscore', 'iqr', 'isolation_forest').
        """
        self.detection_method = method
        features = self.identify_feature_types(df, self.unique_value_threshold)['continuous']

        if method in ['zscore', 'iqr']:
            for feature in features:
                if feature in self.exclude_cols or feature == self.target_column:
                    continue
                self.stats[feature] = self.calculate_feature_stats(df[feature], method)
        elif method == 'isolation_forest':
            self.model = IsolationForest().fit(df[features])
        else:
            raise ValueError(f"Unsupported method: {method}")

    def calculate_feature_stats(self, series, method):
        """
        Calculate necessary statistics for zscore or IQR methods.

        Parameters:
        - series (pd.Series): Data column from the DataFrame.
        - method (str): Specifies which statistics to calculate ('zscore' or 'iqr').

        Returns:
        - dict: Calculated statistics (mean and std for zscore; Q1, Q3, and IQR for IQR).
        """
        if method == 'zscore':
            return {'mean': series.mean(), 'std': series.std()}
        elif method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            return {'Q1': Q1, 'Q3': Q3, 'IQR': IQR}

    def apply_outlier_flags(self, df):
        """
        Apply the fitted model or calculated statistics to identify outliers in any DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame to analyze for outliers.

        Returns:
        - pd.DataFrame: DataFrame with added outlier-related features.
        """
        if not self.detection_method:
            raise ValueError("Detection method has not been set. Please fit the model first.")

        method = self.detection_method
        df_copy = df.copy()

        # Initialize columns to store the results
        df_copy['outlier_count'] = 0
        df_copy['outlier_features'] = [[] for _ in range(len(df))]

        # Assume detect_outliers now returns a DataFrame where True indicates an outlier
        outlier_mask = self.detect_outliers(df_copy)
        
        for feature in outlier_mask.columns:
            for i in df_copy.index:
                if outlier_mask.at[i, feature]:  # If the value is an outlier
                    df_copy.at[i, 'outlier_count'] += 1
                    df_copy.at[i, 'outlier_features'].append(feature)

        # Convert the list of outlier features to a comma-separated string
        df_copy['outlier_features'] = df_copy['outlier_features'].apply(lambda x: ', '.join(x) if x else None)
        
        # Add 'Is_outlier' column based on 'outlier_count'
        df_copy['Is_outlier'] = (df_copy['outlier_count'] > 0).astype(int)
        
        # Map each unique combination of outlier features to a unique ID
        unique_combinations = df_copy['outlier_features'].dropna().unique()
        combination_to_id = {comb: i+1 for i, comb in enumerate(unique_combinations)}
        combination_to_id[None] = 0  # Add a mapping for rows with no outliers
        
        df_copy['outlier_coefficient'] = df_copy['outlier_features'].map(combination_to_id)

        new_df = df_copy.drop('outlier_features', axis=1)
        return new_df

    def detect_outliers(self, df):
        """
        Detect outliers in the DataFrame based on the fitted model or statistics.

        Parameters:
        - df (pd.DataFrame): DataFrame to detect outliers in.

        Returns:
        - pd.DataFrame: DataFrame indicating which observations are outliers.
        """
        features = self.identify_feature_types(df, self.unique_value_threshold)['continuous']
        
        outlier_mask = pd.DataFrame(False, index=df.index, columns=features)

        if self.detection_method in ['zscore', 'iqr']:
            for feature in features:
                if feature in self.exclude_cols or feature == self.target_column:
                    continue
                if self.detection_method == 'zscore':
                    outlier_mask[feature] = self.flag_outliers_z_score(df[feature])
                elif self.detection_method == 'iqr':
                    outlier_mask[feature] = self.flag_outliers_iqr(df[feature])
                elif self.detection_method == 'isolation_forest':
                    outlier_mask[feature] = self.flag_outliers_isolation_forest(df[[feature]])
                elif self.detection_method == 'dbscan':
                    outlier_mask[feature] = self.flag_outliers_dbscan(df[[feature]])
                else:
                    raise ValueError(f"Unsupported method: {self.detection_method}")
        return outlier_mask

    # The higher the threshold, the fewer data points will be classified as outliers.
    def flag_outliers_z_score(self, df, threshold=2):
        """
        Flags outliers based on Z-Score.

        Parameters:
        - df: Series of data points.
        - threshold: Z-Score threshold to use for flagging outliers.

        Returns:
        - A Series indicating outliers (1) and non-outliers (0).
        """
        z_scores = np.abs(stats.zscore(df))
        return (z_scores > threshold).astype(int)
    
    # The higher the threshold, the fewer data points will be classified as outliers.
    def flag_outliers_iqr(self, df, threshold= 3):
        """
        Flags outliers based on the Interquartile Range (IQR).

        Parameters:
        - df: Series of data points.
        - threshold: Multiplier for the IQR to adjust outlier sensitivity.

        Returns:
        - A Series indicating outliers (1) and non-outliers (0).
        """
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        return ((df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR))).astype(int)
    
    def flag_outliers_isolation_forest(self, df, **kwargs):
        """
        Flags outliers using the Isolation Forest method.

        Parameters:
        - df: DataFrame or Series of data points.
        - **kwargs: Additional keyword arguments for the Isolation Forest model.

        Returns:
        - A Series indicating outliers (1) and non-outliers (0).
        """
        model = IsolationForest(**kwargs)
        model.fit(df.values.reshape(-1, 1))
        scores = model.predict(df.values.reshape(-1, 1))
        return (scores == -1).astype(int)

    def flag_outliers_dbscan(self, df, **kwargs):
        """
        Flags outliers using the DBSCAN clustering method.

        Parameters:
        - df: DataFrame or Series of data points.
        - **kwargs: Additional keyword arguments for the DBSCAN model.

        Returns:
        - A Series indicating outliers (1) and non-outliers (0).
        """
        clustering = DBSCAN(**kwargs).fit(df.values.reshape(-1, 1))
        return (clustering.labels_ == -1).astype(int)

    def visualize_outliers(self, df):
        """
        Visualizes outliers in the DataFrame's numeric columns based on the detection method.

        Parameters:
        - df (pd.DataFrame): DataFrame containing the data.
        """
        if not self.detection_method:
            logging.error("Visualization error: Detection method has not been set.")
            return

        features = self.identify_feature_types(df, self.unique_value_threshold)['continuous']
        for feature in features:
            if feature in self.exclude_cols or feature == self.target_column:
                continue
            plt.figure(figsize=(10, 5))
            if self.detection_method in ['zscore', 'iqr']:
                sns.boxplot(x=df[feature], color="lightblue", fliersize=5)
            elif self.detection_method == 'isolation_forest':
                sns.scatterplot(data=df, x=range(len(df)), y=feature, hue='Is_outlier', palette=['green', 'red'], legend=None)
            plt.title(f"{feature} - Outliers by {self.detection_method}")
            plt.xlabel(feature)
            plt.ylabel('Value')
            plt.show()

    def identify_feature_types(self, df, unique_value_threshold):
        """
        Identifies continuous, non-continuous, and binary features in a DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame to analyze.
        - unique_value_threshold (int): Threshold for unique values to consider a feature continuous.

        Returns:
        - dict: Categorized lists of feature names.
        """
        continuous_features, non_continuous_features, binary_features = [], [], []

        for column in df.columns:
            if column in self.exclude_cols or column == self.target_column:
                continue
            if pd.api.types.is_numeric_dtype(df[column]):
                unique_values = df[column].nunique(dropna=False)
                if unique_values == 2:
                    binary_features.append(column)
                elif unique_values > unique_value_threshold:
                    continuous_features.append(column)
                else:
                    non_continuous_features.append(column)
        return {"continuous": continuous_features, 
                "non_continuous": non_continuous_features, 
                "binary": binary_features
                }


# ---------------------- PREPROCESSING: AdvancedAnomalyDetector with autoencoder ---------------------- #

class AdvancedAnomalyDetector:
    """
    Advanced Anomaly Detection Using Autoencoder
    ============================================
    
    This class implements an advanced anomaly detection system using an autoencoder neural network with additional 
    features such as adaptive dropout rates and optional clustering-based thresholding.

    Sections:
    ---------
    - Initialization
    - Data Preprocessing
    - Model Building
    - Model Training
    - Anomaly Detection
    - Utility Methods

    Initialization
    --------------
    
    Initializes the AdvancedAnomalyDetector with the specified parameters.

    Parameters:
    -----------
    - target_column (str): The name of the target column for anomaly detection.
    - exclude_cols (list[str], optional): List of columns to exclude from the feature set.
    - batch_size (int): Batch size for training the autoencoder.
    - epochs (int): Number of epochs to train the autoencoder.
    - validation_split (float): Fraction of the training data to use as validation data.
    - learning_rate (float): Initial learning rate for training the autoencoder.
    - percentile (int): Percentile value to set the anomaly detection threshold. Must be between 1 and 100.

    Data Preprocessing
    ------------------
    
    The `_preprocess_data` method preprocesses the DataFrame by filling in NaN values and extracting features.

    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    --------
    - np.array: The processed feature array ready for training or inference.

    Model Building
    --------------
    
    The `_build_autoencoder` method builds the autoencoder model using the specified number of features.

    Parameters:
    -----------
    - n_features (int): The number of features in the input data.

    Model Training
    --------------
    
    The `fit` method fits the autoencoder model on the provided DataFrame and sets the anomaly detection threshold.

    Parameters:
    -----------
    - df (pd.DataFrame): The training DataFrame.
    - use_clustering_threshold (bool): Whether to use DBSCAN clustering for thresholding.
    - plot_mse_distribution (bool): If True, plot the MSE distribution after fitting.

    Process:
    --------
    1. Preprocesses the data.
    2. Sets the adaptive dropout rate based on the sample size.
    3. Builds the autoencoder model.
    4. Trains the autoencoder with early stopping and learning rate reduction.
    5. Calculates the MSE for the training data.
    6. Sets the anomaly detection threshold based on the specified percentile or using clustering.
    7. Optionally plots the MSE distribution.

    Anomaly Detection
    -----------------
    
    The `_calculate_mse` method calculates the Mean Squared Error (MSE) for each sample based on the autoencoder's predictions.

    Parameters:
    -----------
    - features (np.array): The features array.

    Returns:
    --------
    - np.array: The calculated MSE values.

    The `_find_cluster_threshold` method finds a threshold using clustering (DBSCAN) on MSE values.

    Parameters:
    -----------
    - mse (np.array): The MSE values to cluster.
    - eps (float): The radius around each point to consider as a neighbor.
    - min_samples (int): Minimum number of points to form a cluster.

    Returns:
    --------
    - float: Anomaly threshold determined by DBSCAN.

    Utility Methods
    ---------------
    
    The `_plot_mse_distribution` method plots the MSE distribution and the threshold line if specified.

    Parameters:
    -----------
    - mse (np.array): The MSE values to plot.

    The `predict` method predicts whether each sample in the DataFrame is an anomaly based on the MSE threshold.

    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame to predict anomalies on.

    Returns:
    --------
    - pd.DataFrame: The original DataFrame with an added 'Is_anomaly' column.
    """
    def __init__(self, target_column, exclude_cols=None, 
                batch_size=32, epochs=100, validation_split=0.2, learning_rate=0.001, percentile=97
                ):
        """
        Initializes the AdvancedAnomalyDetector.

        Parameters:
            target_column (str): The name of the target column for anomaly detection.
            exclude_cols (list[str], optional): List of columns to exclude from feature set.
            batch_size (int): Batch size for training the autoencoder.
            epochs (int): Number of epochs to train the autoencoder.
            validation_split (float): Fraction of the training data to use as validation data.
            learning_rate (float): Initial learning rate for training the autoencoder.
            percentile (int): Percentile value to set the anomaly detection threshold. Must be between 1 and 100.
        """
        if not (1 <= percentile <= 100):
            raise ValueError("Percentile must be between 1 and 100.")

        self.target_column = target_column
        self.exclude_cols = exclude_cols if exclude_cols else []
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.percentile = percentile
        self.autoencoder = None
        self.threshold = None
        self.drop_rate = 0.3

    def _preprocess_data(self, df):
        """
        Preprocesses the DataFrame by filling in NaN values and extracting features.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess.

        Returns:
            np.array: The processed feature array ready for training or inference.
        """
        numeric_cols = df.select_dtypes(include=[np.number, "boolean"]).columns.tolist()
        features = df.drop(columns=[self.target_column] + self.exclude_cols, errors='ignore').values.astype(np.float32)
        return features

    def _set_adaptive_drop_rate(self, sample_size):
        """
        Set the dropout rate based on the sample size.

        Parameters:
            sample_size (int): The number of samples in the dataset.
        """
        if sample_size < 1000:
            self.drop_rate = 0.1
        elif sample_size < 10000:
            self.drop_rate = 0.2
        else:
            self.drop_rate = 0.3

    def _build_autoencoder(self, n_features):
        """
        Builds the autoencoder model using the specified number of features.

        Parameters:
            n_features (int): The number of features in the input data.
        """
        encoding_dim = max(1, round(n_features * 0.4))  # Encoding dimension
        input_layer = layers.Input(shape=(n_features,))

        # Encoder network
        x = layers.Dense(encoding_dim * 2)(input_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(self.drop_rate)(x)

        x = layers.Dense(encoding_dim)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(self.drop_rate)(x)

        # Decoder network
        x = layers.Dense(encoding_dim * 2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        decoded = layers.Dense(n_features, activation='sigmoid')(x)

        self.autoencoder = models.Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, df, use_clustering_threshold=False, plot_mse_distribution=True):
        """
        Model Training
        --------------
        
        Fits the autoencoder model on the provided DataFrame and sets the anomaly detection threshold.

        Parameters:
        -----------
        - df (pd.DataFrame): The training DataFrame.
        - use_clustering_threshold (bool): Whether to use DBSCAN clustering for thresholding.
        - plot_mse_distribution (bool): If True, plot the MSE distribution after fitting.

        Process:
        --------
        1. Preprocesses the data.
        2. Sets the adaptive dropout rate based on the sample size.
        3. Builds the autoencoder model.
        4. Trains the autoencoder with early stopping and learning rate reduction.
        5. Calculates the MSE for the training data.
        6. Sets the anomaly detection threshold based on the specified percentile or using clustering.
        7. Optionally plots the MSE distribution.

        Preprocesses the data and sets up the model for training. The `fit` method follows these steps:
        1. **Data Preprocessing:** Preprocess the input DataFrame to extract the features.
        2. **Adaptive Dropout Rate:** Set the dropout rate adaptively based on the sample size.
        3. **Build Autoencoder:** Build the autoencoder model architecture.
        4. **Train Autoencoder:** Train the autoencoder model using the training data with early stopping and learning rate reduction.
        5. **Calculate MSE:** Calculate the Mean Squared Error (MSE) for the training data.
        6. **Set Threshold:** Set the anomaly detection threshold based on the specified percentile or using DBSCAN clustering.
        7. **Plot MSE Distribution:** Optionally plot the MSE distribution to visualize the threshold.

        Raises:
        -------
        - ValueError: If the specified percentile is not between 1 and 100.

        Usage:
        ------
        >>> detector = AdvancedAnomalyDetector(target_column='target', epochs=50, percentile=95)
        >>> detector.fit(train_df, plot_mse_distribution=True)
        """
        features = self._preprocess_data(df)
        self._set_adaptive_drop_rate(len(features))
        self._build_autoencoder(features.shape[1])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10)
        
        # Train the autoencoder
        self.autoencoder.fit(
            features, features,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping, reduce_lr]
            )
        logging.info("Autoencoder training completed.")

        mse = self._calculate_mse(features)

        if use_clustering_threshold:
            self.threshold = self._find_cluster_threshold(mse)
        else:
            # Set threshold using the adjusted percentile
            self.threshold = np.percentile(mse, self.percentile)   # Calculate the percentile as threshold

        logging.info(f"Anomaly threshold set at: {self.threshold:.2f}")

        if plot_mse_distribution:
            self._plot_mse_distribution(mse)

    def _calculate_mse(self, features):
        """
        Calculates the Mean Squared Error (MSE) for each sample based on the autoencoder's predictions.

        Parameters:
            features (np.array): The features array.

        Returns:
            np.array: The calculated MSE values.
        """
        predictions = self.autoencoder.predict(features)
        mse = np.mean(np.power(features - predictions, 2), axis=1)
        return mse

    def _find_cluster_threshold(self, mse, eps=0.4, min_samples=10):
        """
        Find a threshold using clustering (DBSCAN) on MSE values.

        Parameters:
            mse (np.array): The MSE values to cluster.
            eps (float): The radius around each point to consider as a neighbor.
            min_samples (int): Minimum number of points to form a cluster.

        Returns:
            float: Anomaly threshold determined by DBSCAN.
        """
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(mse.reshape(-1, 1))
        noise = mse[clustering.labels_ == -1]  # Label '-1' indicates noise
        return np.min(noise) if noise.size > 0 else np.percentile(mse, 99)

    def _plot_mse_distribution(self, mse):
        """
        Plots the MSE distribution and the threshold line if specified.

        Parameters:
            mse (np.array): The MSE values to plot.
        """
        plt.figure(figsize=(8, 5))
        ax = sns.histplot(mse, bins=50, kde=True, fill=True, color="#D6DAFF")
        ax.lines[0].set_color('#17354C')

        plt.title('Distribution of MSE Values')
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Frequency')
        plt.axvline(self.threshold, color='#230FA0', linestyle='--', linewidth=2)

        # Calculate y-position for the text to be centered vertically
        y_position = plt.ylim()[1] * 0.8

        # Add text with white background centered on the vertical line
        plt.text(self.threshold, y_position, f'Percentile:\n{self.percentile:.0f} %',
                color='#230FA0', ha='center', va='center', 
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round, pad=0.3'))
        plt.show()

    def predict(self, df):
        """
        Predicts whether each sample in the DataFrame is an anomaly based on the MSE threshold.
        -------------------------------------------------------------------------------------

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to predict anomalies on.

        Returns
        -------
        pd.DataFrame
            The original DataFrame with an added 'is_anomaly' column.

        Raises
        ------
        Exception
            If the model has not been fitted yet.
        """
        if self.autoencoder is None or self.threshold is None:
            raise Exception("Model not fitted. Call fit() with training data before predict().")

        features = self._preprocess_data(df)
        mse = self._calculate_mse(features)
        df['Is_anomaly'] = (mse > self.threshold).astype(int)
        print(pd.Series(mse).describe())
        return df

# ---------------------- PREPROCESSING: AnomalyDetector with autoencoder ---------------------- #

class AnomalyDetector:
    """
    Anomaly Detection Using Autoencoder
    ===================================
    
    This class implements an anomaly detection system using an autoencoder neural network. The model is trained to 
    reconstruct normal data, and anomalies are detected based on reconstruction error.

    Sections:
    ---------
    - Initialization
    - Data Preprocessing
    - Model Building
    - Model Training
    - Anomaly Detection
    - Utility Methods

    Initialization
    --------------
    
    Initializes the AnomalyDetector with the specified parameters.

    Parameters:
    -----------
    - target_column (str): The name of the target column for anomaly detection.
    - exclude_cols (list[str], optional): List of columns to exclude from the feature set.
    - batch_size (int): Batch size for training the autoencoder.
    - epochs (int): Number of epochs to train the autoencoder.
    - validation_split (float): Fraction of the training data to use as validation data.
    - percentile (int): Percentile value to set the anomaly detection threshold. Must be between 1 and 100.

    Data Preprocessing
    ------------------
    
    The `_preprocess_data` method preprocesses the DataFrame by filling in NaN values and extracting features.

    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    --------
    - np.array: The processed feature array ready for training or inference.

    Model Building
    --------------
    
    The `_build_autoencoder` method builds the autoencoder model using the specified number of features.

    Parameters:
    -----------
    - n_features (int): The number of features in the input data.

    Model Training
    --------------
    
    The `fit` method fits the autoencoder model on the provided DataFrame and sets the anomaly detection threshold.

    Parameters:
    -----------
    - df (pd.DataFrame): The training DataFrame.
    - plot_mse_distribution (bool): If True, plot the MSE distribution after fitting.

    Process:
    --------
    1. Preprocesses the data.
    2. Builds the autoencoder model.
    3. Trains the autoencoder with early stopping.
    4. Calculates the MSE for the training data.
    5. Sets the anomaly detection threshold based on the specified percentile.
    6. Optionally plots the MSE distribution.

    Anomaly Detection
    -----------------
    
    The `_calculate_mse` method calculates the Mean Squared Error (MSE) for each sample based on the autoencoder's predictions.

    Parameters:
    -----------
    - features (np.array): The features array.

    Returns:
    --------
    - np.array: The calculated MSE values.

    Utility Methods
    ---------------
    
    The `_plot_mse_distribution` method plots the MSE distribution and the threshold line if specified.

    Parameters:
    -----------
    - mse (np.array): The MSE values to plot.

    The `predict` method predicts whether each sample in the DataFrame is an anomaly based on the MSE threshold.

    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame to predict anomalies on.

    Returns:
    --------
    - pd.DataFrame: The original DataFrame with an added 'Is_anomaly' column.
    """
    def __init__(self, target_column, exclude_cols=None, 
                batch_size=32, epochs=100, validation_split=0.2, percentile=97
                ):

        if not (1 <= percentile <= 100):
            raise ValueError("Percentile must be between 1 and 100.")

        self.target_column = target_column
        self.exclude_cols = exclude_cols if exclude_cols else []
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.percentile = percentile
        self.autoencoder = None
        self.threshold = None

    def _preprocess_data(self, df):
        """
        Preprocesses the DataFrame by filling in NaN values and extracting features.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess.

        Returns:
            np.array: The processed feature array ready for training or inference.
        """
        numeric_cols = df.select_dtypes(include=[np.number, "boolean"]).columns.tolist()
        #df[numeric_cols] = df[numeric_cols].fillna(0)  # Replace NaNs with zeros
        # Select features, excluding the target column and any columns to exclude
        features = df.drop(columns=[self.target_column] + self.exclude_cols, errors='ignore').values.astype(np.float32)
        return features

    def _build_autoencoder(self, n_features):
        """
        Builds the autoencoder model using the specified number of features.

        Parameters:
            n_features (int): The number of features in the input data.
        """
        encoding_dim = max(1, round(n_features * 0.3))  # Calculate encoding dimension
        input_layer = layers.Input(shape=(n_features,))
        encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
        decoded = layers.Dense(n_features, activation='sigmoid')(encoded)
        self.autoencoder = models.Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    def fit(self, df, plot_mse_distribution=True):
        """
        Model Training
        --------------
        
        Fits the autoencoder model on the provided DataFrame and sets the anomaly detection threshold.

        Parameters:
        -----------
        - df (pd.DataFrame): The training DataFrame.
        - plot_mse_distribution (bool): If True, plot the MSE distribution after fitting.

        Process:
        --------
        1. Preprocesses the data.
        2. Builds the autoencoder model.
        3. Trains the autoencoder with early stopping.
        4. Calculates the MSE for the training data.
        5. Sets the anomaly detection threshold based on the specified percentile.
        6. Optionally plots the MSE distribution.
        """
        features = self._preprocess_data(df)
        self._build_autoencoder(features.shape[1])
        
        # Train the autoencoder
        self.autoencoder.fit(
                             features, features,
                             epochs=self.epochs,
                             batch_size=self.batch_size,
                             validation_split=self.validation_split,
                             callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5)]
                             )
        logging.info("Autoencoder training completed.")

        mse = self._calculate_mse(features)
        self.threshold = np.percentile(mse, self.percentile)  # Calculate the percentile as threshold

        logging.info(f"Anomaly threshold set at: {self.threshold:.2f}")

        if plot_mse_distribution:
            self._plot_mse_distribution(mse)

    def _calculate_mse(self, features):
        """
        Calculates the Mean Squared Error (MSE) for each sample based on the autoencoder's predictions.

        Parameters:
            features (np.array): The features array.

        Returns:
            np.array: The calculated MSE values.
        """
        predictions = self.autoencoder.predict(features)
        mse = np.mean(np.power(features - predictions, 2), axis=1)
        return mse
    
    def _plot_mse_distribution(self, mse):
        """
        Plots the MSE distribution and the threshold line if specified.

        Parameters:
            mse (np.array): The MSE values to plot.
            plot_mse_distribution (bool): Flag indicating whether to plot the distribution.
        """
        plt.figure(figsize=(8, 5))
        ax=sns.histplot(mse, bins=50, kde=True, fill=True, color="#D6DAFF")
        ax.lines[0].set_color('#17354C')
        plt.title('Distribution of MSE Values')
        plt.xlabel('Mean Squared Error')
        plt.ylabel('Frequency')
        plt.axvline(self.threshold, color='#230FA0', linestyle='--', linewidth=2)
        #plt.text(self.threshold + 0.01, plt.ylim()[1] * 0.9, f'Threshold: {self.threshold:.2f}', color='#230FA0')
        
        # Calculate y-position for the text to be centered vertically
        y_position = plt.ylim()[1] * 0.8

        # Add text with white background centered on the vertical line
        plt.text(self.threshold, y_position, f'Percentile: {self.percentile:.0f} %',
                color='#230FA0', ha='center', va='center', 
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))

        plt.show()

    def predict(self, df):
        """
        Predicts whether each sample in the DataFrame is an anomaly based on the MSE threshold.
        -------------------------------------------------------------------------------------

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to predict anomalies on.

        Returns
        -------
        pd.DataFrame
            The original DataFrame with an added 'is_anomaly' column.

        Raises
        ------
        Exception
            If the model has not been fitted yet.
        """
        if self.autoencoder is None or self.threshold is None:
            raise Exception("Model not fitted. Call fit() with training data before predict().")

        features = self._preprocess_data(df)
        mse = self._calculate_mse(features)
        df['Is_anomaly'] = (mse > self.threshold).astype(int)  # Assign 1 for anomalies, 0 for normal
        print(pd.Series(mse).describe())
        return df

# ---------------------- PREPROCESSING: Handling Class Imbalance -------------------- #

def handle_class_imbalance(df, 
                           dataset_name='Train', 
                           target_column=None, 
                           unique_id_col=None, 
                           threshold=0.5, 
                           random_state=42):
    """
    Balance class distribution
    ==========================

    Balances the class distribution in a dataset using SMOTE (Synthetic Minority Over-sampling Technique).
    The unique identifier column (if provided) is excluded from the resampling process and later re-associated.

    Parameters:
    -----------

    - df (pd.DataFrame): DataFrame containing the features and the target column.
    - dataset_name (str): A descriptive name for the dataset being processed, used in logging and print statements.
    - target_column (str): The name of the target column. This column should exist in `df`.
    - unique_id_col (str, optional): The name of the unique identifier column (e.g., 'client_id'). This column, if provided, will be excluded from resampling. Default is None.
    - threshold (float): The threshold for applying SMOTE. Determines the resampling strategy proportion. Default is 0.5.
    - random_state (int): A seed used by the random number generator for SMOTE, ensuring reproducibility. Default is 42.

    Returns:
    --------
    - pd.DataFrame: A DataFrame with a balanced class distribution based on the specified `target_column`.

    Raises:
    -------
    - ValueError: If the `target_column` or `unique_id_col` (if provided) is not found in the DataFrame.
    """
    
    # Ensure target column exists in the DataFrame
    if target_column not in df.columns:
        raise ValueError(f"'{target_column}' not found in dataframe columns.")

    # Ensure unique_id_col exists if provided
    if unique_id_col and unique_id_col not in df.columns:
        raise ValueError(f"'{unique_id_col}' not found in dataframe columns.")
    
    # Separate the unique_id_col if provided
    client_ids = None
    if unique_id_col:
        client_ids = df[unique_id_col]
        df = df.drop(columns=[unique_id_col])

    # Splitting features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Create DataFrame for class distribution and print it
    class_distribution = pd.DataFrame(y.value_counts()).reset_index()
    class_distribution.columns = ['Target', 'Count of Uniques']
    print(f"\nOriginal Distribution of {dataset_name} Dataset:")
    print(class_distribution.to_string(index=False, header=True))
    print("\n" + "-"*50)

    # Proportion threshold to determine imbalance. 
    threshold_imbalanced=0.5

    # Check if significant imbalance exists and apply SMOTE if needed
    if (class_distribution['Count of Uniques'] / class_distribution['Count of Uniques'].sum()).max() >= threshold_imbalanced:
        logging.info(f"SMOTE applied on '{dataset_name}' dataset.")

        # Applying SMOTE
        smote = SMOTE(sampling_strategy=threshold, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Combining resampled features and target
        df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

        # Re-attach client_ids to original rows, NaN to synthetic rows
        if client_ids is not None:
            df_resampled[unique_id_col] = client_ids.reindex(df_resampled.index, fill_value=pd.NA)

        # Print resampled distribution
        resampled_distribution = pd.DataFrame(y_resampled.value_counts()).reset_index()
        resampled_distribution.columns = ['Target', 'Count of Uniques']
        print(f"\nResampled Distribution of {dataset_name} Dataset:")
        print(resampled_distribution.to_string(index=False, header=True))
        print("\n" + "-"*50)

        return df_resampled
    else:
        logging.info(f"The dataset '{dataset_name}' is fairly balanced. SMOTE not applied.")
        return df

# ---------------------- HELPER FUNCTION : Validate & prepare data ---------------------- #

def validate_and_prepare(df, target_column, exclude_cols=None):
    """
    Validation of dataframe
    =======================

    Validates the input DataFrame and prepares it for model training by excluding specified columns
    and ensuring only numeric feature columns are included.

    This function performs several checks to ensure the input DataFrame is suitable for further processing
    and model training. It dynamically handles the exclusion of specified columns, the target column, and
    the filtering out of non-numeric columns to prevent issues during model training.

    Parameters:
    -----------
    - df (pd.DataFrame): The input DataFrame containing features and a target column.
    - target_column (str): The name of the target column in the DataFrame.
    - exclude_cols (list of str or str, optional): Column names to exclude from the feature set. Can be a list of strings or a single string.

    Returns:
    --------
    - pd.DataFrame: A DataFrame containing only numeric feature columns after exclusions.
    - list: The list of feature column names included in the returned DataFrame.

    Raises:
    -------
    - TypeError: If the input `df` is not a pandas DataFrame.
    - ValueError: If `df` is empty, the target column is not found, or specified columns to exclude are not found.
    """
    # Input DataFrame validation
    if not isinstance(df, pd.DataFrame):
        logging.error("The provided data is not a pandas DataFrame.")
        raise TypeError("Input data must be a pandas DataFrame.")
    if df.empty:
        logging.error("The provided DataFrame is empty.")
        raise ValueError("Input DataFrame is empty.")
    if target_column not in df.columns:
        logging.error(f"Target column '{target_column}' not found in the DataFrame.")
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")
    # Handling the exclusion of columns
    if exclude_cols:
        if isinstance(exclude_cols, str):
            exclude_cols = [exclude_cols]
        if not all(col in df.columns for col in exclude_cols):
            missing_cols = [col for col in exclude_cols if col not in df.columns]
            logging.error(f"Columns to exclude not found in the DataFrame: {missing_cols}")
            raise ValueError(f"One or more columns to exclude are not in the DataFrame: {missing_cols}")
    else:
        exclude_cols = []
    
    # Prepare data
    excluded_columns = set([target_column] + exclude_cols)
    feature_columns = [col for col in df.columns if col not in excluded_columns]

    # Identify and excluding non-numeric columns
    non_numeric_cols = df[feature_columns].select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        logging.info(f"Found non-numeric columns: {non_numeric_cols}. Excluding.")
        feature_columns = [col for col in feature_columns if col not in non_numeric_cols]

    # Ensuring there are features left for training
    if not feature_columns:
        logging.error("No numeric feature columns available for model training.")
        raise ValueError("No numeric feature columns available after processing.")
    #logging.info(f"Numeric Features selected for model training: {feature_columns}")
    
    return df[feature_columns], feature_columns


# ---------------------- DefinitionExtractor Class : Extract top-level function and class definitions ---------------------- #
class DefinitionExtractor:
    """
    Definition Extraction & Documentation Display
    ==============================================
    
    This class provides methods to extract top-level function and class definitions from Python source files and 
    dynamically display their documentation in a Jupyter notebook environment. It includes capabilities for extracting 
    definitions, collecting top-level callables, and setting up interactive widgets for documentation display.

    Sections:
    ---------
    - Initialization
    - Definition Extraction
    - Callable Collection
    - Documentation Display
    - Utility Methods

    Initialization
    --------------
    
    Initializes the DefinitionExtractor with a predefined list of utility files.

    Parameters:
    -----------
    None

    Definition Extraction
    ---------------------
    
    The `extract_definitions` method extracts top-level function and class names from a specified Python source file.

    Parameters:
    -----------
    - file_path (str): The path to the Python source file.

    Returns:
    --------
    - List[str]: A list of top-level function and class names defined in the file.

    Callable Collection
    -------------------
    
    The `get_top_level_callables` method retrieves top-level callables from a module based on extracted definitions.

    Parameters:
    -----------
    - module (Any): The module from which to retrieve callables.
    - definitions (List[str]): A list of names of the top-level functions and classes.

    Returns:
    --------
    - Dict[str, Any]: A dictionary mapping callable names to callable objects.

    Documentation Display
    ---------------------
    
    The `setup_dropdown_widget` method sets up and displays a dropdown widget for selecting and displaying documentation.

    Examples:
    ---------
    >>> de = DefinitionExtractor()
    >>> de.setup_dropdown_widget()

    Utility Methods
    ---------------
    
    - `_collect_definitions`: Collects and stores all top-level function and class names from the specified utility files.
    - `_collect_top_level_callables`: Collects and stores top-level callables from the specified utility files.

    Example Usage:
    --------------
    >>> de = DefinitionExtractor()
    >>> de.setup_dropdown_widget()

    Methods:
    --------
    """
    def __init__(self):
        self.utility_files = ['common_utils.py', 'model_utils.py', 'plotting_utils.py']
        self.excluded_names = ['adjust_lightness', 'analyze_dataframe', 'calculate_and_display_metrics', 
                               'calculate_lift', 'calculate_metric', 'categorical_analysis_conv',
                               'check_date_columns', 'check_duplicates', 'check_missing_values',
                               'check_unique_values', 'compare_dataframes', 'compare_roc_curves',
                               'compare_train_val_metrics', 'compare_y_val_arrays', 'create_predictions', 'DefinitionExtractor', 
                               'detect_outliers_conv', 'display_evaluation_metrics', 'encode_target', 'evaluate_model',
                               'evaluate_predictions', 'evaluate_threshold_metrics', 'find_elbow_point',
                               'generate_correlation_heatmap_conv', 'lgbm_custom_eval', 'load_data', 'load_widget_state', 'logreg_custom_eval',
                               'merge_datasets', 'panel_of_confusion_matrices', 'perform_fold_logreg', 'plot_advanced_learning_curves', 
                               'plot_comparison_chart',
                               'plot_cumulative_gain', 'plot_evals_results', 'plot_feature_importance', 
                               'plot_multiple_roc_prc', 'plot_performance_summary', 'predict_with_model',
                               'remove_duplicates', 'rename_and_compare_columns', 'save_df', 'shap_analysis', 'split_data',
                               'train_model_logreg', 'unique_values', 'versatile_custom_eval','xgb_custom_eval'
                               ] 

        self.utility_functions = {}
        self._collect_definitions()
        self._collect_top_level_callables()
        self.setup_dropdown_widget()

    def __repr__(self):
        """
        Override the default representation of the class instance to prevent unwanted output.
        """
        return ""

    def extract_definitions(self, file_path: str) -> List[str]:
        """
        Extract function and class definitions from a Python source file, excluding methods.

        Parameters:
        -----------
        file_path : str
            The path to the Python source file.

        Returns:
        --------
        list
            A list of top-level function and class names defined in the file.
        """
        definitions = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                stripped_line = line.strip()
                if (stripped_line.startswith('def ') or stripped_line.startswith('class ')) and not line.startswith(' '):
                    match = re.match(r'^(def|class) (\w+)', stripped_line)
                    if match and match.group(2) not in self.excluded_names:
                        definitions.append(match.group(2))
        return definitions

    def _collect_definitions(self) -> None:
        """
        Collects and stores all top-level function and class names from the specified utility files.
        """
        all_definitions = []
        for file_path in self.utility_files:
            all_definitions.extend(self.extract_definitions(file_path))
        self.all_definitions = all_definitions

    def get_top_level_callables(self, module: Any, definitions: List[str]) -> Dict[str, Any]:
        """
        Get top-level callables from a module based on extracted definitions.

        Parameters:
        -----------
        module : Any
            The module from which to retrieve callables.
        definitions : List[str]
            A list of names of the top-level functions and classes.

        Returns:
        --------
        Dict[str, Any]
            A dictionary mapping callable names to callable objects.
        """
        callables = {}
        for name in definitions:
            member = getattr(module, name, None)
            if callable(member) and not hasattr(member, '__self__'):
                callables[name] = member
        return callables

    def _collect_top_level_callables(self) -> None:
        """
        Collects and stores top-level callables from the specified utility files.
        """
        for file_path in self.utility_files:
            module_name = file_path.replace('.py', '')
            module = importlib.import_module(module_name)
            callables = self.get_top_level_callables(module, self.all_definitions)
            self.utility_functions.update(callables)

    def setup_dropdown_widget(self) -> None:
        """
        Sets up and displays a dropdown widget for selecting and displaying documentation.

        The dropdown widget is populated with the names of the top-level functions and classes
        extracted from the specified utility files. When a name is selected, the corresponding
        documentation is displayed below the dropdown.
        """
        sorted_keys = sorted([key for key in self.utility_functions.keys() if key not in self.excluded_names], key=lambda s: s.lower())
        
        func_dropdown = widgets.Dropdown(
            options=['Select function or class'] + sorted_keys,
            description='Select:',
            disabled=False,
        )

        doc_output = widgets.Output()

        def display_doc(change):
            if change['new'] != 'Select function or class':
                selected_func = self.utility_functions[change['new']]
                doc_output.clear_output()
                with doc_output:
                    doc_content = f"### {change['new']} Documentation\n\n{selected_func.__doc__}"
                    display(Markdown(doc_content))

        func_dropdown.observe(display_doc, names='value')

        display(func_dropdown, doc_output)
