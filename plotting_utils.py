# ---------------------- Execute save dataframes function & print results ---------------------- #
# ---------------------- Standard Library Imports ---------------------- #
import sys
import logging
import math
import warnings
from itertools import cycle

# ---------------------- Data Handling and Processing ---------------------- #
import numpy as np
import pandas as pd

# ---------------------- Data Visualization ---------------------- #
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap, to_rgba, to_hex, to_rgb
import colorsys
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
from plotly.offline import iplot, plot, init_notebook_mode
import seaborn as sns
from tabulate import tabulate

# Initialize Plotly to run offline in a Jupyter Notebook
init_notebook_mode(connected=True)

# ---------------------- General Machine Learning Utilities ---------------------- #
import shap  # Model interpretability

# ---------------------- Data Preprocessing and Feature Engineering ---------------------- #
from sklearn.preprocessing import StandardScaler

# ---------------------- Model Training and Evaluation ---------------------- #
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, make_scorer, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
)

# ---------------------- IMPORT MY_MODULES  ---------------------- #
import model_utils as mu
import common_utils as cu

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

# Set display options for Pandas
pd.set_option('display.width', 1000)



# ---------------------- Exploratory Data Analysis (EDA): MULTIVARIATE ANALYSIS ----------------- #

class DataAnalyzer:
    """
    Data Analysis and Correlation Insight Class
    ===========================================

    This class provides advanced data analysis capabilities focusing on identifying correlations and generating
    insights between features in a DataFrame, specifically tailored around a target variable.

    Sections:
    ---------
    - Initialization
    - Data Preparation
    - Target Encoding
    - Correlation Analysis
    - Insight Generation
    - Visualization
    - Utility Methods

    Initialization
    --------------

    Initializes the DataAnalyzer with the dataset and analysis parameters.

    Parameters:
    -----------
    - df (pd.DataFrame): The dataset to analyze.
    - target_column (str, optional): The primary column of interest for correlation analysis.
    - exclude_cols (list[str], optional): Columns to exclude from analysis to avoid skewing results.
    - corr_threshold_strong (float, optional): Threshold above which a correlation is considered strong.
    - corr_threshold_weak (float, optional): Threshold below which a correlation is considered weak.
    - multicollinearity_threshold (float, optional): Threshold above which correlations are noted for multicollinearity issues.

    Data Preparation
    ----------------

    The `prepare_data` method prepares the data by removing specified columns and normalizing numeric features.

    Method:
    -------
    - `prepare_data`: Removes specified columns and normalizes numeric features.

    Parameters:
    -----------
    - None

    Raises:
    -------
    - KeyError: If the specified target column is not found in the DataFrame.

    Target Encoding
    ---------------

    The `encode_target` method encodes the target column if it is categorical but not binary, raising an error if conditions are not met.

    Method:
    -------
    - `encode_target`: Encodes the target column if categorical and binary.

    Parameters:
    -----------
    - None

    Raises:
    -------
    - ValueError: If the target column is categorical but not binary.

    Correlation Analysis
    --------------------

    The `find_highly_correlated_pairs` method identifies pairs of variables with correlation coefficients above a specified threshold.

    Method:
    -------
    - `find_highly_correlated_pairs`: Identifies highly correlated variable pairs.

    Parameters:
    -----------
    - corr_matrix (pd.DataFrame): The correlation matrix of the DataFrame.

    Returns:
    --------
    - dict: Pairs of variables with high correlation.

    Insight Generation
    ------------------

    The `get_insight_for_feature` method generates insights based on the correlation strength and thresholds.

    Method:
    -------
    - `get_insight_for_feature`: Generates insights for a specific feature.

    Parameters:
    -----------
    - feature (str): The feature for which to generate insights.
    - corr_matrix (pd.DataFrame): The correlation matrix of the DataFrame.

    Returns:
    --------
    - str: Insights based on correlation thresholds.

    Visualization
    -------------

    The `generate_correlation_heatmap` method generates a heatmap for the correlations of specified features.

    Method:
    -------
    - `generate_correlation_heatmap`: Generates a correlation heatmap.

    Parameters:
    -----------
    - features_df (pd.DataFrame): DataFrame containing features to include in the heatmap.
    - threshold (float, optional): Threshold for displaying correlation annotations.
    - figsize (tuple, optional): Size of the heatmap figure.

    Raises:
    -------
    - Warning: If no data is available for heatmap generation due to feature filtering.

    Utility Methods
    ---------------

    The `generate_insights` method generates insights from the DataFrame based on the specified correlation thresholds.

    Method:
    -------
    - `generate_insights`: Generates and returns insights for the dataset.

    Parameters:
    -----------
    - None

    Returns:
    --------
    - tuple: A DataFrame with insights and a DataFrame with highly correlated features.

    Example Usage:
    --------------
    >>> analyzer = DataAnalyzer(df, target_column='target', exclude_cols=['id'], 
                                corr_threshold_strong=0.5, corr_threshold_weak=0.1, 
                                multicollinearity_threshold=0.75)
    >>> insights, highly_corr_df = analyzer.generate_insights()
    >>> analyzer.generate_correlation_heatmap(highly_corr_df)
    """
    def __init__(self, df, target_column=None, exclude_cols=None, 
                 corr_threshold_strong=0.5, corr_threshold_weak=0.1, 
                 multicollinearity_threshold=0.75):
        """
        Initializes the DataAnalyzer with the dataset and analysis parameters.

        Parameters:
            df (pd.DataFrame): The dataset to analyze.
            target_column (str): The primary column of interest for correlation analysis.
            exclude_cols (list[str], optional): Columns to be excluded from analysis to avoid skewing results.
            corr_threshold_strong (float): Threshold above which a correlation is considered strong.
            corr_threshold_weak (float): Threshold below which a correlation is considered weak.
            multicollinearity_threshold (float): Threshold above which correlations are noted for multicollinearity issues.
        """
        self.df = df
        self.target_column = target_column
        self.exclude_cols = exclude_cols
        self.corr_threshold_strong = corr_threshold_strong
        self.corr_threshold_weak = corr_threshold_weak
        self.multicollinearity_threshold = multicollinearity_threshold
        self.prepare_data()

    def prepare_data(self):
        """Prepares the data by removing specified columns and normalizing numeric features."""
        if self.exclude_cols:
            self.df = self.df.drop(columns=[col for col in self.exclude_cols if col in self.df.columns], errors='ignore')
        
        if self.target_column and self.target_column not in self.df.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in DataFrame columns.")

        # Normalize numeric columns
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            scaler = StandardScaler()
            self.df[numeric_columns] = scaler.fit_transform(self.df[numeric_columns])
        
        if self.target_column and (self.df[self.target_column].dtype == 'object' or pd.api.types.is_categorical_dtype(self.df[self.target_column])):
            self.encode_target()

    def encode_target(self):
        """Encodes the target column if it is categorical but not binary, raising an error if conditions are not met."""
        if self.df[self.target_column].nunique() == 2:
            self.df[self.target_column] = pd.Categorical(self.df[self.target_column]).codes
            logging.info(f"Target '{self.target_column}' encoded.")
        else:
            logging.error("Target column is categorical but not binary, cannot encode.")
            raise ValueError("Target column must be binary to auto-encode.")

    def find_highly_correlated_pairs(self, corr_matrix):
        """Identifies pairs of variables with correlation coefficients above a specified threshold."""
        highly_correlated_pairs = {}
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > self.multicollinearity_threshold:
                    var1, var2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    highly_correlated_pairs.setdefault(var1, []).append(var2)
                    highly_correlated_pairs.setdefault(var2, []).append(var1)
        return highly_correlated_pairs

    def get_insight_for_feature(self, feature, corr_matrix):
        """Generates insights based on the correlation strength and thresholds."""
        insights = []
        target_corr = corr_matrix.at[feature, self.target_column] if self.target_column else None

        if pd.isna(target_corr):
            return 'Constant'
        
        if target_corr is not None:
            if abs(target_corr) >= self.corr_threshold_strong:
                insights.append('Strong Correlation with Target')
            if abs(target_corr) <= self.corr_threshold_weak:
                insights.append('Lack of Correlation')
            if self.corr_threshold_weak < abs(target_corr) < self.corr_threshold_strong:
                insights.append('Moderate Correlation')
        
        if any(abs(corr_matrix[feature].drop([feature] + ([self.target_column] if self.target_column else []))) > self.multicollinearity_threshold):
            insights.append('Multicollinearity')

        return ', '.join(insights) if insights else 'No specific insight'

    def generate_correlation_heatmap(self, features_df, threshold=0.01, figsize=(16, 14)):
        """Generates a heatmap for the correlations of specified features."""
        if features_df is None or features_df.empty:
            logging.warning("No data available for heatmap generation due to feature filtering.")
            return  # Skip plotting the heatmap if no data available

        features_to_include = features_df['Feature'].tolist() + ([self.target_column] if self.target_column else [])
        filtered_corr_matrix = self.df[features_to_include].corr()

        mask = np.triu(np.ones_like(filtered_corr_matrix, dtype=bool))
        annotation_mask = (np.abs(filtered_corr_matrix) > threshold) & ~mask
        annotations = np.where(annotation_mask, filtered_corr_matrix.round(2).astype(str), "")

        extreme_colors = ["#643EF0", "white", "#E1FF58"]
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", extreme_colors, N=100)

        plt.figure(figsize=figsize)
        sns.heatmap(filtered_corr_matrix, mask=mask, annot=annotations, cmap=custom_cmap, center=0,
                    vmin=-1, vmax=1, cbar_kws={"shrink": 0.3}, fmt='', annot_kws={"size": 12})
        plt.title('Correlation Heatmap of Selected Features')
        plt.show()

    def generate_insights(self):
        """Generates insights from the DataFrame based on the specified correlation thresholds."""
        non_numeric_columns = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_columns:
            logging.info(f"Dropping non-numeric columns: {non_numeric_columns}")

        df_numeric = self.df.select_dtypes(include=[np.number])

        if df_numeric.empty:
            raise ValueError("No numeric columns available for correlation analysis after processing.")

        corr_matrix = df_numeric.corr()
        if corr_matrix.empty:
            raise ValueError("Correlation matrix computation failed; possibly due to insufficient data.")

        correlated_pairs = self.find_highly_correlated_pairs(corr_matrix)
        if self.target_column:
            insights = pd.DataFrame(corr_matrix[self.target_column].sort_values(ascending=False)).reset_index()
        else:
            # Average the correlations across all columns if no target is specified
            insights = pd.DataFrame(corr_matrix.mean(axis=1)).reset_index()
        insights.columns = ['Feature', 'Correlation']
        if self.target_column:
            insights = insights[insights['Feature'] != self.target_column]
        insights['Insight'] = insights['Feature'].apply(lambda x: self.get_insight_for_feature(x, corr_matrix))
        insights['Multicollinearity with'] = insights['Feature'].apply(lambda x: ', '.join(correlated_pairs.get(x, [])))

        constants = insights[insights['Insight'].str.contains('Constant')]
        if not constants.empty:
            logging.info(f"Features with no variation (constant): {constants['Feature'].tolist()}")
            insights = insights[~insights['Insight'].str.contains('Constant')]

        highly_corr_df = insights[~(
            (insights['Correlation'] > -0.3) &
            (insights['Correlation'] < 0.3) &
            (insights['Insight'] == 'Lack of Correlation') &
            (insights['Multicollinearity with'] == '')
        )]

        if highly_corr_df.empty:
            logging.warning("No features meet the criteria for the correlation heatmap.")
        else:
            self.generate_correlation_heatmap(highly_corr_df)

        return insights, highly_corr_df


# ---------------------- Exploratory Data Analysis (EDA): MULTIPLE CATEGORICAL STACK ANALYSIS ----------------- #

class CategoricalPlot:
    """
    Multiple Categorical Stack Analysis
    =====================================
    
    This class creates visually appealing stacked bar plots of categorical data using Plotly. It helps in understanding the distribution
    and proportion of different categories within the dataset features.

    Sections:
    ---------
    - Initialization
    - Color Utilities
    - Plotting
    - Utility Methods

    Initialization
    --------------

    Initializes the CategoricalPlot class with the provided DataFrame, dataset name, and an optional list of columns to exclude.

    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame containing the data.
    - dataset_name (str, optional): A label for the dataset, used in the plot title.
    - exclude_cols (list[str] or str or None, optional): Columns to exclude from plotting.

    Color Utilities
    ---------------

    Utility methods for color manipulation and interpolation.

    Methods:
    --------
    - `is_light_color`: Determines if the given color is light or dark based on its hex code.
    - `interpolate_colors`: Generates a list of n colors, interpolating between two hex colors.

    Plotting
    --------

    Generates and displays a stacked bar plot of categorical data. It automatically excludes specified columns and calculates the
    percentage distributions of categories within each feature.

    Methods:
    --------
    - `plot`: Generates and displays the stacked bar plot.

    Utility Methods
    ---------------

    - `is_light_color`: Determines if the given color is light or dark based on its hex code.
    - `interpolate_colors`: Generates a list of n colors, interpolating between two hex colors.

    Example Usage:
    --------------
    >>> plotter = CategoricalPlot(df, dataset_name='Sample Dataset', exclude_cols=['id'])
    >>> plotter.plot()

    Methods:
    --------
    """
    def __init__(self, df, dataset_name='Dataset', exclude_cols=None):
        """
        Initializes the CategoricalPlot class with the provided DataFrame,
        dataset name, and an optional list of columns to exclude.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            dataset_name (str): A label for the dataset, used in the plot title.
            exclude_cols (list[str] or str or None): Columns to exclude from plotting.
        """
        self.df = df
        self.dataset_name = dataset_name
        self.exclude_cols = [exclude_cols] if isinstance(exclude_cols, str) else exclude_cols
        init_notebook_mode(connected=True)  # Initialize Plotly for offline mode
        self.plot()  # Automatically plot on instance creation

    def is_light_color(self, hex_color):
        """
        Determines if the given color is light or dark based on its hex code.
        Light colors are defined as having a luminance greater than 0.5.

        Parameters:
            hex_color (str): The hexadecimal color code.

        Returns:
            bool: True if the color is light, False if dark.
        """
        rgb = to_rgb(hex_color)
        luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
        return luminance > 0.5

    def interpolate_colors(self, color1, color2, n):
        """
        Generates a list of n colors, interpolating between two hex colors.

        Parameters:
            color1 (str): Start color in hexadecimal.
            color2 (str): End color in hexadecimal.
            n (int): Number of colors to generate.

        Returns:
            list[str]: List of interpolated hexadecimal colors.
        """
        colors = [to_rgb(color1), to_rgb(color2)]
        return [to_hex([(colors[0][j] + (colors[1][j] - colors[0][j]) * i / (n - 1)) for j in range(3)]) for i in range(n)]

    def plot(self):
        """
        Generates and displays a stacked bar plot of categorical data.
        It automatically excludes specified columns and calculates the
        percentage distributions of categories within each feature.
        """
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("The df parameter must be a pandas DataFrame.")
        
        if self.df.empty:
            print("The DataFrame is empty. No data to plot.")
            return None
        
        base_colors = ["#E1FF58", "#643EF0"]  # Base colors for two unique categories
        
        if self.exclude_cols is not None:
            if isinstance(self.exclude_cols, str):
                self.exclude_cols = [self.exclude_cols]
            elif not isinstance(self.exclude_cols, list):
                raise ValueError("exclude_cols should be a string or a list of strings.")
        else:
            self.exclude_cols = []

        categorical_features = self.df.select_dtypes(include=['object', 'category']).columns
        categorical_features = [feature for feature in categorical_features if feature not in self.exclude_cols]

        if not categorical_features:
            print("No categorical columns to plot.")
            return

        fig = go.Figure()

        for feature in categorical_features:
            counts = self.df[feature].value_counts().sort_index()
            percentages = counts / counts.sum()
            n_values = len(counts)
            
            colors = self.interpolate_colors(base_colors[0], base_colors[1], n_values) if n_values > 2 else base_colors
            
            cumulative_percentage = 0
            for i, (value, count) in enumerate(counts.items()):
                percentage_value = percentages[value] * 100
                cumulative_percentage += percentage_value
                color = colors[i if i < n_values else -1]
                bar_is_light = self.is_light_color(color)
                
                fig.add_trace(go.Bar(
                    name=str(value),
                    x=[percentage_value],
                    y=[feature],
                    orientation='h',
                    marker=dict(color=color),
                    hoverinfo='text',
                    hovertext=f'Value: {value}<br>Count: {count}<br>Percentage: {percentage_value:.2f}%',
                    text=f'{count}',
                    textposition='inside',
                    insidetextanchor='middle',
                    textfont=dict(color='white' if not bar_is_light else 'black'),
                ))

        fig.update_layout(
            barmode='stack',
            title=f'Stacked Barplot of Categorical Features - {self.dataset_name}',
            plot_bgcolor='white',
            paper_bgcolor='white',

            xaxis=dict(
                title='Percentage',
                tickfont=dict(size=10),
                tickvals=list(range(0, 101, 10)),
                range=[-1, 101],
                showgrid=True,
                gridcolor='#DDDDDD',
                gridwidth=0.5,
            ),
            yaxis=dict(
                title='Feature',
                tickfont=dict(size=10),
                automargin=True
            ),
            showlegend=False,
            height=600,
            width=1000
        )

        #fig.show()
        iplot(fig)  # Use iplot to display the figure inline


# ---------------------- Exploratory Data Analysis (EDA): UNIVARIATE ANALYSIS (numerical) ----------------- #

class NumericAnalysis:
    """
    Univariate Analysis of Numerical Features
    =========================================
    
    This class provides functionality for performing univariate analysis on numerical features of a given DataFrame.
    It includes methods to adjust the lightness of a given color and to plot distributions of numerical features.

    **Sections:**
    -------------
    - Initialization
    - Color Adjustment
    - Analysis
    - Utility Methods

    **Initialization**
    ------------------
    Initializes the NumericAnalysis class with the provided DataFrame, dataset name, and variance threshold.

    **Parameters:**
    ---------------
    - `df (pd.DataFrame)`: The DataFrame on which to perform univariate analysis.
    - `dataset_name (str, optional)`: A name for the dataset, used in the title of plots. Default is 'Dataset'.
    - `variance_threshold (float, optional)`: The variance threshold to filter numerical features. Default is 0.01.

    **Color Adjustment**
    --------------------
    Utility methods for adjusting the lightness of colors.

    **Methods:**
    ------------
    - `adjust_lightness(color, amount=0.5)`: Adjusts the lightness of the given color.

    **Analysis**
    ------------
    Performs univariate analysis on the numerical features of the DataFrame, generating distribution plots.

    **Methods:**
    ------------
    - `perform_analysis()`: Generates distribution plots and provides a summary report for numerical features.

    **Utility Methods**
    -------------------
    - `adjust_lightness(color, amount=0.5)`: Adjusts the lightness of the given color.

    **Example Usage:**
    ------------------
    ```python
    >>> analyzer = NumericAnalysis(df, dataset_name='Sample Dataset', variance_threshold=0.01)
    >>> fig = analyzer.perform_analysis()
    ```

    **Methods:**
    ------------
    """
    def __init__(self, df, dataset_name='Dataset', variance_threshold=0.01):
        """
        Initializes the NumericAnalysis class with the provided DataFrame, dataset name, and variance threshold.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame on which to perform univariate analysis.
        dataset_name : str, optional
            A name for the dataset, used in the title of plots. Default is 'Dataset'.
        variance_threshold : float, optional
            The variance threshold to filter numerical features. Default is 0.01.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The df parameter must be a pandas DataFrame.")
        
        self.df = df
        self.dataset_name = dataset_name
        self.variance_threshold = variance_threshold
        self.base_color = "#643EF0"
        self.perform_analysis() 
    
    @staticmethod
    def adjust_lightness(color, amount=0.5):
        """
        Adjust the lightness of the given color.

        Parameters:
        -----------
        color : str
            Hex code of the base color.
        amount : float, optional
            Factor by which the lightness is adjusted, greater than 1 to lighten, less than 1 to darken.
            Default is 0.5.
        
        Returns:
        --------
        str
            New color code after adjusting the lightness.
        """
        try:
            # Convert the hex color code to an RGBA tuple and then to HLS (Hue, Lightness, Saturation)
            c = colorsys.rgb_to_hls(*to_rgba(color)[:3])

            # Adjust the lightness: multiply the current lightness by the given amount
            new_lightness = max(0, min(1, c[1] * amount))

            # Convert the adjusted HLS value back to RGB and then to hex color code
            return to_hex(colorsys.hls_to_rgb(c[0], new_lightness, c[2]))
        except Exception as e:
            # Log an error message if any exception occurs
            logging.error(f"Failed to adjust lightness of color {color} - Error: {e}")

            # Return the original color if adjustment fails
            return color

    def perform_analysis(self):
        """
        Perform univariate analysis on the numerical features of the DataFrame.

        This function generates distribution plots for numerical features and provides 
        a summary report for numerical features.

        Returns:
        --------
        matplotlib.figure.Figure
            A matplotlib figure containing the plots.
        
        Raises:
        -------
        ValueError
            If the input is not a pandas DataFrame.
        """
        numerical_features = self.df.select_dtypes(include=['number']).columns
        numerical_features = [feature for feature in numerical_features if self.df[feature].var() >= self.variance_threshold]

        num_plots = len(numerical_features)
        cols = 3
        rows = (num_plots + cols - 1) // cols
        figsize = (7 * cols, 6 * rows)

        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
        axes = axes.flatten()

        skipped_features = []

        for feature in numerical_features:
            feature_variance = self.df[feature].var()
            if feature_variance < self.variance_threshold:
                skipped_features.append(f'{feature} (variance: {feature_variance})')

        if skipped_features:
            logging.info(f'Skipping features with variance lower than {self.variance_threshold}: ' + ', '.join(skipped_features))

        for idx, feature in enumerate(numerical_features):
            ax = axes[idx]

            if feature in skipped_features:
                continue

            counts, bins, patches = ax.hist(self.df[feature], bins=25, color=self.base_color, zorder=3)
            max_count = max(counts)
            min_count = min(counts)
            count_range = max_count - min_count
            if count_range == 0:
                count_range = 1

            for count, patch in zip(counts, patches):
                lightness = 1 - (count - min_count) / count_range
                lightness = lightness ** 0.8

                # Increase base_lightness to make the lightest color lighter
                # Decrease base_lightness to make the lightest color darker
                base_lightness = 0.2  # Adjust this value as needed to make the lightest color lighter

                lightness = base_lightness + (1 - base_lightness) * lightness
                patch.set_facecolor(self.adjust_lightness(self.base_color, lightness))

            ax.set_title(f"Distribution of '{feature}'", pad=11)
            ax.set_xlabel(feature, fontsize=10, labelpad=11)
            ax.set_ylabel('Frequency' if feature in numerical_features else 'Count', fontsize=10)
            ax.grid(True, zorder=0)

        for ax in axes[num_plots:]:
            ax.set_visible(False)

        fig.tight_layout(pad=3.0)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        plt.suptitle(f'Univariate Numeric Analysis of {self.dataset_name}', fontsize=14, fontweight='bold', y=1.003)
        plt.show()
        return fig


# ---------------------- Exploratory Data Analysis (EDA): Uniques per numerical features ----------------- #

def plot_unique_values(df, dataset_name='Dataset', exclude_cols=None, log_scale=True):
    """
    Plotly bar chart
    =================
    Plots the number of unique values per numerical feature, with options to exclude columns, set log scale, and adjust font size.
    
    **Parameters:**
    ----------------
    - df (pd.DataFrame): The DataFrame containing the data.
    - dataset_name (str, optional): The name of the dataset, used for the plot title. Defaults to 'Dataset'.
    - exclude_cols (list, optional): List of column names to exclude from the analysis. Defaults to None.
    - log_scale (bool, optional): If True, the y-axis will use a logarithmic scale. Defaults to True.
    
    **Returns:**
    ------------
    - plotly.graph_objs.Figure: Plotly bar chart showing the unique value counts per numerical feature.

    **Raises:**
    -----------
    - TypeError: If the `df` parameter is not a pandas DataFrame or if `exclude_cols` is not a list or None.
    - KeyError: If any column in `exclude_cols` is not present in the DataFrame.
    """
    # Type checks
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The `df` parameter must be a pandas DataFrame.")
    if exclude_cols is not None and not isinstance(exclude_cols, list):
        raise TypeError("The `exclude_cols` parameter must be a list or None.")
    
    # Check that all excluded columns are in the DataFrame
    if exclude_cols:
        missing_cols = [col for col in exclude_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {', '.join(missing_cols)}")

    # Filter out the excluded columns and select numerical features
    numerical_features = df.select_dtypes(include="number").columns
    if exclude_cols:
        numerical_features = [col for col in numerical_features if col not in exclude_cols]

    # Compute the number of unique values per numerical feature
    unique_counts = df[numerical_features].nunique().sort_values()

    color="#643EF0" 
    x_axis_label_size=10

    # Initialize Plotly for offline use
    init_notebook_mode(connected=True)

    # Create Plotly bar chart with custom hover text
    fig = go.Figure(data=[go.Bar(
        x=unique_counts.index,
        y=unique_counts.values,
        marker=dict(color=color),
        hovertemplate="Value: %{x},<br>Count: %{y}<extra></extra>",  # Custom hover template
    )])

    # Configure layout settings with x-axis label size
    fig.update_layout(
        title=f"Unique Values Per Numerical Feature in {dataset_name}",
        xaxis_title="Feature",
        yaxis_title="Number of Unique Values",
        yaxis_type="log" if log_scale else "linear",
        xaxis=dict(
            tickangle=-90,  # Rotate x-axis labels by 90 degrees
            tickfont=dict(size=x_axis_label_size)  # Set font size of x-axis labels
        ),
        template="plotly_white",
        autosize=True,
        height=600,
        width=1000,
        #margin=dict(l=40, r=40, t=60, b=80),
    )
    #fig.show()
    iplot(fig)  # Use iplot to display the figure inline        


# ---------------------- Exploratory Data Analysis (EDA): Outliers Detection (plotly) ----------------- #

def detect_outliers(df, dataset_name='Dataset', exclude_cols=None, variance_threshold=0.03, iqr_threshold=0.1, log_scale=False):
    """
    Outlier Detection and Visualization
    ===================================
    
    This function generates box plots and scatter plots for all numerical features in the DataFrame to detect outliers. It is designed to
    help identify and visualize outliers within numerical data by providing a clear representation of the distribution and variance of each feature.

    Sections:
    ---------
    - Initialization
    - Data Preparation
    - Feature Filtering
    - Plot Generation
    - Usage

    Initialization
    --------------
    Initializes the function with the provided DataFrame, dataset name, and optional parameters for outlier detection.

    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame on which to perform univariate analysis.
    - dataset_name (str, optional): A name for the dataset, used in the title of plots. Defaults to 'Dataset'.
    - exclude_cols (list, optional): List of column names to exclude from the analysis. Defaults to None.
    - variance_threshold (float, optional): The variance threshold to filter numerical features. Defaults to 0.03.
    - iqr_threshold (float, optional): The interquartile range threshold to filter numerical features. Defaults to 0.1.
    - log_scale (bool, optional): If True, the y-axis will use a logarithmic scale. Defaults to False.

    Data Preparation
    ----------------
    Prepares the data by checking for necessary conditions and excluding specified columns.

    Raises:
    -------
    - TypeError: If `df` is not a pandas DataFrame or `exclude_cols` is not a list.
    - KeyError: If any columns specified in `exclude_cols` are not found in the DataFrame.

    Feature Filtering
    -----------------
    Filters out numerical features with low variance, low IQR, or insufficient unique values.

    Plot Generation
    ---------------
    Generates box plots and scatter plots for the filtered features to visualize outliers.

    Example Usage:
    --------------
    >>> detect_outliers(df, dataset_name='Sample Dataset', exclude_cols=['id'], variance_threshold=0.03, iqr_threshold=0.1, log_scale=True)

    Returns:
    --------
    - Plotly figure object if successful. 
    """
    # Type checks
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The `df` parameter must be a pandas DataFrame.")
    if df.empty:
        logging.error("The DataFrame is empty. No data to plot.")
        return None
    if exclude_cols is not None and not isinstance(exclude_cols, list):
        raise TypeError("The `exclude_cols` parameter must be a list or None.")
    
    # Check that all excluded columns are in the DataFrame
    if exclude_cols:
        missing_cols = [col for col in exclude_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {', '.join(missing_cols)}")

    base_color = "#643EF0"
    line_color = "#FE00C9"
    
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Filter out numerical features with low variance, low IQR, or insufficient unique values
    filtered_features = []
    for feature in numerical_features:
        if df[feature].var() >= variance_threshold and df[feature].nunique() > 2:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > iqr_threshold:
                filtered_features.append(feature)
    
    if not filtered_features:
        logging.error("No numeric columns with sufficient variance, IQR, and unique values found in the DataFrame.")
        return None

    skipped_features = []  # List to hold the names of skipped features
    # Loop to determine which features to skip
    for feature in numerical_features:
        feature_variance = df[feature].var()
        if feature_variance < variance_threshold or df[feature].nunique() <= 2:
            # Add the feature name to the skipped_features list
            skipped_features.append(f'{feature} (variance: {feature_variance}, unique values: {df[feature].nunique()})')
    # Log skipped features before plotting
    if skipped_features:
        logging.info(f' Skipping features with variance lower than {variance_threshold}: ' + ', '.join(skipped_features))
    
    # Prepare data for plots
    # Prepare data for plots
    data_to_plot = df[filtered_features]
    data_melted = data_to_plot.melt(var_name='Feature', value_name='Value')

    # Create an array of rainbow colors
    num_colors = len(filtered_features)
    rainbow_colors = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, num_colors)]

    # Initialize Plotly for offline use
    init_notebook_mode(connected=True)

    # Create a Plotly figure
    fig = go.Figure()

    # Create box plots with unique rainbow colors for each feature
    for feature, color in zip(filtered_features, rainbow_colors):
        feature_data = data_melted[data_melted['Feature'] == feature]['Value']
        
        # Box plot
        fig.add_trace(go.Box(
            y=feature_data,
            name=feature,
            boxpoints=False,   # 'outliers',
            marker=dict(
                color=color,  # Use a rainbow color for each box plot
                line=dict(
                    outliercolor=line_color,
                    outlierwidth=2)),
            line=dict(color=color, width=2),  # Outline color
            opacity=0.7
        ))

    # Configure the layout to match the desired style
    fig.update_layout(
        title=f'Outlier Analysis of {dataset_name}',
        xaxis=dict(
            title='Features',
            tickangle=-90,
            tickfont=dict(size=10),
            categoryorder='array',
            categoryarray=filtered_features
        ),
        yaxis=dict(
            title='Numerical Value',
            tickfont=dict(size=10),
            type='log' if log_scale else 'linear'
        ),
        plot_bgcolor='#EFF1FB',
        margin=dict(l=40, r=40, b=120, t=80),
        showlegend=False  # Hide the entire legend
    )
    #fig.show()
    iplot(fig) # Use iplot to display the figure inline


# ---------------------- STRIP CHARTS: DISTRIBUTION & PATTERNS ---------------------- #

def strip_chart(df, dataset_name='Dataset', exclude_cols=None, variance_threshold=0.03, iqr_threshold=0.1, log_scale=False):
    """
    Strip Chart Generation for Outlier Detection
    ============================================
    
    This function generates strip charts for all numerical features in the DataFrame to detect outliers. It is designed to help
    identify and visualize outliers within numerical data by providing a clear representation of the distribution and variance
    of each feature.

    Sections:
    ---------
    - Initialization
    - Data Preparation
    - Methods
    - Usage

    Initialization
    --------------
    Initializes the function with the provided DataFrame, dataset name, and optional parameters for outlier detection.

    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame on which to perform univariate analysis.
    - dataset_name (str, optional): A name for the dataset, used in the title of plots. Defaults to 'Dataset'.
    - exclude_cols (list, optional): List of column names to exclude from the analysis. Defaults to None.
    - variance_threshold (float, optional): The variance threshold to filter numerical features. Defaults to 0.03.
    - iqr_threshold (float, optional): The interquartile range threshold to filter numerical features. Defaults to 0.1.
    - log_scale (bool, optional): If True, the y-axis will use a logarithmic scale. Defaults to False.

    Data Preparation
    ----------------
    Prepares the data by checking for necessary conditions and excluding specified columns.

    Raises:
    -------
    - TypeError: If `df` is not a pandas DataFrame or `exclude_cols` is not a list.
    - KeyError: If any columns specified in `exclude_cols` are not found in the DataFrame.

    Methods:
    --------
    - `filter_features`: Identifies and retains features meeting the specified thresholds for variance and IQR.
    - `generate_plots`: Creates and configures Plotly visualizations.

    Example Usage:
    --------------
    >>> strip_chart(df, dataset_name='Sample Dataset', exclude_cols=['id'], variance_threshold=0.03, iqr_threshold=0.1, log_scale=True)

    Returns:
    --------
    - Plotly figure object if successful. 
    """
    # Type checks
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The `df` parameter must be a pandas DataFrame.")
    if df.empty:
        logging.error("The DataFrame is empty. No data to plot.")
        return None
    if exclude_cols is not None and not isinstance(exclude_cols, list):
        raise TypeError("The `exclude_cols` parameter must be a list or None.")
    
    # Check that all excluded columns are in the DataFrame
    if exclude_cols:
        missing_cols = [col for col in exclude_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {', '.join(missing_cols)}")

    numerical_features = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
    
    # Filter out numerical features with low variance, low IQR, or insufficient unique values
    filtered_features = []
    for feature in numerical_features:
        if df[feature].var() >= variance_threshold and df[feature].nunique() > 2:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > iqr_threshold:
                filtered_features.append(feature)
    
    if not filtered_features:
        logging.error("No numeric columns with sufficient variance, IQR, and unique values found in the DataFrame.")
        return None

    skipped_features = []  # List to hold the names of skipped features
    # Loop to determine which features to skip
    for feature in numerical_features:
        feature_variance = df[feature].var()
        if feature_variance < variance_threshold or df[feature].nunique() <= 2:
            # Add the feature name to the skipped_features list
            skipped_features.append(f'{feature} (variance: {feature_variance}, unique values: {df[feature].nunique()})')
    # Log skipped features before plotting
    if skipped_features:
        logging.info(f' Skipping features with variance lower than {variance_threshold}: ' + ', '.join(skipped_features))
    
    # Prepare data for strip charts
    data_to_plot = df[filtered_features]
    data_melted = data_to_plot.melt(var_name='Feature', value_name='Value')


    # Create an array of rainbow colors
    num_colors = len(filtered_features)
    rainbow_colors = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, num_colors)]

    # Initialize Plotly for offline use
    init_notebook_mode(connected=True)

    # Create a Plotly figure
    fig = go.Figure()

    # Create box plots with unique rainbow colors for each feature
    for feature, color in zip(filtered_features, rainbow_colors):
        feature_data = data_melted[data_melted['Feature'] == feature]['Value']
        
        
        # Scatter points directly with the feature names as x-axis labels
        scatter_x = [feature] * len(feature_data)
        fig.add_trace(go.Scatter(
            x=scatter_x,
            y=feature_data,
            mode='markers',
            name="",  # Set name to an empty string to hide from hover/legend
            marker=dict(size=2, color=color, opacity=0.4),  # Adjusted opacity and size for smaller dots
            hovertemplate="Feature: %{x},<br>Value: %{y}<extra></extra>",  # Display feature name and y-value correctly
            showlegend=False
        ))

    # Configure the layout to match the desired style
    fig.update_layout(
        title=f'Distribution of {dataset_name}',
        xaxis=dict(
            title='Features',
            tickangle=-90,  # Rotate labels to 90 degrees
            tickfont=dict(size=10),  # Adjust the font size
            categoryorder='array',  # Preserve the feature order
            categoryarray=filtered_features
        ),
        yaxis=dict(
            title='Numerical Value',
            tickfont=dict(size=10),
            type='log' if log_scale else 'linear'  # Toggle logarithmic scale based on the parameter
        ),
        plot_bgcolor='#EFF1FB',  # Set to a light background color matching the uploaded image
        margin=dict(l=40, r=40, b=120, t=80)
    )
    #fig.show()
    iplot(fig)     # Use iplot to display the figure inline


# ---------------------- EDA: Distribution of specific categorical feature ----------------- #

def create_pie_chart(df, dataset_name='Dataset', categorical_feature=None):
    """
    Pie Chart Creation for Categorical Feature Analysis
    ===================================================
    
    This function creates a pie chart for a specific categorical feature in the given DataFrame. It is designed to help
    visualize the distribution of categories within a feature, providing an intuitive understanding of the categorical data.

    Sections:
    ---------
    - Initialization
    - Data Preparation
    - Plot Generation
    - Error Handling

    Initialization
    --------------

    Initializes the function with the provided DataFrame, dataset name, and categorical feature.

    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame containing the data.
    - dataset_name (str, optional): The name of the dataset for the title of the chart. Defaults to 'Dataset'.
    - categorical_feature (str, optional): The name of the categorical feature to plot.

    Data Preparation
    ----------------

    Prepares the data by validating the input parameters and converting numeric features to categorical if necessary.

    Raises:
    -------
    - TypeError: If `df` is not a pandas DataFrame or `categorical_feature` is not specified.
    - KeyError: If the specified `categorical_feature` is not found in the DataFrame.

    Plot Generation
    ---------------

    Generates a pie chart for the specified categorical feature.

    Methods:
    --------
    - `generate_pie_chart`: Creates and configures the pie chart using Plotly.

    Error Handling
    --------------

    Catches and logs errors that occur during the pie chart creation process.

    Example Usage:
    --------------
    >>> create_pie_chart(df, dataset_name='Sample Dataset', categorical_feature='Category')

    Returns:
    --------
    - Plotly figure object if successful. None if not successful.
    """
    # Validate input parameters
    if categorical_feature is None:
        logging.info("No categorical feature specified.")
        return None
    
    if categorical_feature not in df.columns:
        logging.info(f"The feature '{categorical_feature}' is not in the DataFrame columns.")
        return None

    # Prepare the feature data for plotting
    feature_data = df[categorical_feature]

    # Check the data type of the feature
    if feature_data.dtype in [int, float]:
        # Convert the numeric feature to string for categorical plotting
        feature_data = feature_data.astype(str)
        logging.info(f"The feature '{categorical_feature}' is numeric. Temporarily converted to categorical for plotting.")
    elif feature_data.dtype.name == 'category':
        logging.info(f"The feature '{categorical_feature}' is already a category. No conversion needed.")
    else:
        # Assume any other type is treatable as categorical
        logging.info(f"The feature '{categorical_feature}' is treated as categorical.")

    # Initialize Plotly for offline use
    init_notebook_mode(connected=True)

    # Plotting
    try:
        counts = feature_data.value_counts().reset_index()
        counts.columns = [categorical_feature, 'count']  # Renaming the columns appropriately

        custom_colors = ['#643EF0', '#E1FF58', '#17354C', '#BBE4F2', '#269DAA', '#596DE1', '#82CCFF',
                         '#230FA0', '#FF6692', '#40225D', '#9CF9FE', '#40225D']

        fig = px.pie(
            counts,
            names=categorical_feature,
            values='count',
            title=f"Distribution of '{categorical_feature}' in {dataset_name}",
            color_discrete_sequence=custom_colors,
            hole=0.3,
            labels={'count': 'Count'}
        )
        # Custom hover data
        fig.update_traces(
            textinfo='percent+label',
            hoverinfo='label+percent+name',
            hovertemplate='<b>Value: %{label}</b><br>Count: %{value}<br>Percentage: %{percent:.1%}<extra></extra>',
            textfont={'size': 16}  # increases the font size for better readability
        )

        fig.update_layout(
            legend_title_text=categorical_feature,
            height=600,
            width=900
        )
        #fig.show()
        # Use iplot to display the figure inline
        iplot(fig)

    except Exception as e:
        logging.error(f"An error occurred while creating the pie chart: {e}")
        return None



# ---------------------- Exploratory Data Analysis (EDA): UNIVARIATE ANALYSIS (categorical & conventional)  ----------------- #
def adjust_lightness(color, amount=0.5):
    """
    Adjust the lightness of the given color.

    Parameters:
    -----------
    color : str
        Hex code of the base color.
    amount : float, optional
        Factor by which the lightness is adjusted, greater than 1 to lighten, less than 1 to darken.
        Default is 0.5.
    
    Returns:
    --------
    str
        New color code after adjusting the lightness.
    """
    try:
        # Convert the hex color code to an RGBA tuple and then to HLS (Hue, Lightness, Saturation)
        c = colorsys.rgb_to_hls(*to_rgba(color)[:3])

        # Adjust the lightness: multiply the current lightness by the given amount
        new_lightness = max(0, min(1, c[1] * amount))

        # Convert the adjusted HLS value back to RGB and then to hex color code
        return to_hex(colorsys.hls_to_rgb(c[0], new_lightness, c[2]))
    except Exception as e:
        # Log an error message if any exception occurs
        logging.error(f"Failed to adjust lightness of color {color} - Error: {e}")

        # Return the original color if adjustment fails
        return color
        
def categorical_analysis_conv(df, dataset_name='Dataset', max_unique_categories=25, exclude_cols=None):
    """
    Perform univariate analysis on the given DataFrame, focusing on categorical variables.
    Generates bar charts for categorical features with controlled aesthetics, excluding specified columns.

    Parameters:
    - df (pd.DataFrame): The DataFrame on which to perform univariate analysis.
    - dataset_name (str): A name for the dataset, used in the title of plots.
    - max_unique_categories (int): The maximum number of unique categories for which to plot distributions.
    - exclude_cols (str or list): Columns to exclude from the analysis.

    Returns:
    - plots (list): A list of matplotlib figure objects containing the plots.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The df parameter must be a pandas DataFrame.")

    # Handle the exclude_cols parameter to ensure it's iterable
    if exclude_cols is not None:
        if isinstance(exclude_cols, str):
            exclude_cols = [exclude_cols]  # Make a list if only one column name is provided
        elif not isinstance(exclude_cols, list):
            raise ValueError("exclude_cols should be a string or a list of strings.")
    else:
        exclude_cols = []

    base_color = "#643EF0"

    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    categorical_features = [feature for feature in categorical_features if feature not in exclude_cols]

    if len(categorical_features) == 0:
        logging.info("No suitable categorical columns found in the DataFrame.")
        return

    # Determine the layout of the subplots
    num_plots = sum(df[feature].nunique() <= max_unique_categories for feature in categorical_features)
    cols = 2  # Number of columns
    rows = (num_plots + 1) // cols  # Calculate rows needed
    
    # Dynamically adjust figure size based on the number of plots
    figsize = (8 * cols, 6 * rows)

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)  # Adjust the figure size as necessary
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    bar_width = 0.6

    for idx, feature in enumerate(categorical_features):
        unique_categories = df[feature].nunique()
        ax = axes[idx]

        if unique_categories <= max_unique_categories:
            count_data = df[feature].value_counts().sort_values(ascending=False)
            
            # Calculate the positions for the bars
            positions = range(len(count_data))
            # Plot the bars directly using matplotlib
            ax.bar(positions, count_data.values, width=bar_width, color=base_color, zorder=3)

            # Now set the custom tick labels
            ax.set_xticks(positions)
            ax.set_xticklabels(count_data.index, rotation=45)

            # Normalize counts to get the scaling factor for lightness adjustment
            max_count = count_data.max()
            min_count = count_data.min()
            bars = ax.patches  # Get the bars created by plt.bar
            for bar, count in zip(bars, count_data.values):
                # Adjust lightness: more counts, darker color; fewer counts, lighter color
                lightness = 1 - 0.5 * (count - min_count) / (max_count - min_count) if max_count != min_count else 0.5
                bar.set_facecolor(adjust_lightness(base_color, lightness))

            # Set title, labels, and grid
            ax.set_title(f"Distribution of '{feature}'", pad=11)
            ax.set_xlabel(feature, fontsize=10, labelpad=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.grid(True, zorder=0)

            ax.tick_params(axis='x', labelsize=8, labelrotation=45)
            ax.tick_params(axis='y', labelsize=8)
            ax.set_axisbelow(True)
            ax.grid(True)

        else:
            logging.info(f'Skipping {feature} as it exceeds the max_unique_categories limit.')

    # Hide any unused axes
    for ax in axes[num_plots:]:
        ax.set_visible(False)

    # Adjust layout to prevent overlap with suptitle
    fig.tight_layout(pad=3.0)

    # Adjust the spacing between the subplots
    fig.subplots_adjust(hspace=0.5, wspace=0.3)  
    
    # Add the overall title before the subplots
    plt.suptitle(f'Univariate Categorical Analysis of {dataset_name}', fontsize=14, fontweight='bold', y=1.01)
    plt.show()

    return fig  # Return the figure containing all subplots


# ---------------------- Exploratory Data Analysis (EDA): CORRELATION HEATMAP (conventional) ----------------- #

def encode_target(df, target_column):
    """
    Encodes a binary target column if it's categorical to enable correlation analysis.
    
    Parameters:
    -----------
    - df (pd.DataFrame): DataFrame containing the target column.
    - target_column (str): Name of the column to encode.
    
    Returns:
    --------
    - pd.DataFrame: Modified DataFrame with the target column encoded if it was categorical.
    
    Raises:
    --------    
    - ValueError: If the target column is categorical but not binary.
    """
    if df[target_column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[target_column]):
        if df[target_column].nunique() == 2:
            df[target_column] = pd.Categorical(df[target_column]).codes
            logging.info(f"Target '{target_column}' encoded.")
        else:
            logging.error("Target column is categorical but not binary, cannot encode.")
            raise ValueError("Target column must be binary to auto-encode.")
    return df

def generate_correlation_heatmap_conv(df, target_column, features_df=None, threshold=0.3, figsize=(16, 14)):
    """
    Generates a correlation heatmap for specified features within a DataFrame.
    Ensures all features and the target are numeric to compute correlations.

    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame containing the data.
    - target_column (str): The name of the target column.
    - features_df (pd.DataFrame): DataFrame containing the names of features to include in the heatmap.
    - threshold (float): Threshold for displaying annotations in the heatmap.
    - figsize (tuple): The dimensions for the figure size.

    Raises:
    --------  
    - ValueError: If necessary conditions on data are not met.
    """
    if features_df is None or features_df.empty:
        logging.error("Features DataFrame is empty or not provided.")
        raise ValueError("A non-empty DataFrame containing feature names must be provided.")

    if target_column not in df.columns:
        logging.error(f"The target column '{target_column}' does not exist in the DataFrame.")
        raise ValueError(f"The target column '{target_column}' must be present in the DataFrame.")

    # Ensure all features and target are numeric
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        logging.info(f"Target column '{target_column}' is not numeric. Attempting to encode...")
        df = encode_target(df, target_column)

    features_to_include = features_df['Feature'].tolist() + [target_column]
    non_numeric_features = [feature for feature in features_to_include if not pd.api.types.is_numeric_dtype(df[feature])]

    if non_numeric_features:
        logging.error(f"Non-numeric features found: {non_numeric_features}. Cannot compute correlation.")
        raise ValueError("All features must be numeric to compute correlation.")

    filtered_corr_matrix = df[features_to_include].corr()

    mask = np.triu(np.ones_like(filtered_corr_matrix, dtype=bool))
    annotation_mask = (np.abs(filtered_corr_matrix) > threshold) & ~mask
    annotations = np.where(annotation_mask, filtered_corr_matrix.round(2).astype(str), "")

    # Define the color map
    extreme_colors = ["#643EF0", "white", "#E1FF58"]  # Colors for -1, 0, +1
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", extreme_colors, N=100)

    plt.figure(figsize=figsize)
    sns.heatmap(filtered_corr_matrix, mask=mask, annot=annotations, cmap=custom_cmap, center=0,
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.3}, fmt='', annot_kws={"size": 12})
    plt.title('Correlation Heatmap of Selected Features')
    plt.show()


# ---------------------- Exploratory Data Analysis (EDA): Outliers Detection (conventional) ----------------- #

def detect_outliers_conv(df, dataset_name='Dataset', variance_threshold=0.03, iqr_threshold=0.1):
    """
    Generate box plots for all numerical features in the DataFrame to detect outliers.
    
    Parameters:
    -----------
    - df (pd.DataFrame): The DataFrame on which to perform univariate analysis.
    - dataset_name (str): A name for the dataset, used in the title of plots.
    - variance_threshold (float): The variance threshold to filter numerical features.
    - iqr_threshold (float): The interquartile range threshold to filter numerical features.
    
    Returns:
    ---------
    list: A list of matplotlib figure objects containing the box plots, if any numeric columns are found.
    None:
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The df parameter must be a pandas DataFrame.")
    
    if df.empty:
        logging.error("The DataFrame is empty. No data to plot.")
        return None
    
    base_color = "#643EF0"
    
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Filter out numerical features with low variance, low IQR, or insufficient unique values
    filtered_features = []
    for feature in numerical_features:
        if df[feature].var() >= variance_threshold and df[feature].nunique() > 2:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > iqr_threshold:
                filtered_features.append(feature)
    
    if not filtered_features:
        logging.error("No numeric columns with sufficient variance, IQR, and unique values found in the DataFrame.")
        return None

    skipped_features = []  # List to hold the names of skipped features
    # Loop to determine which features to skip
    for feature in numerical_features:
        feature_variance = df[feature].var()
        if feature_variance < variance_threshold or df[feature].nunique() <= 2:
            # Add the feature name to the skipped_features list
            skipped_features.append(f'{feature} (variance: {feature_variance}, unique values: {df[feature].nunique()})')
    # Log skipped features before plotting
    if skipped_features:
        logging.info(f' Skipping features with variance lower than {variance_threshold}: ' + ', '.join(skipped_features))

    # Prepare data for boxplot
    data_to_plot = df[filtered_features]
    data_melted = data_to_plot.melt(var_name='Feature', value_name='Value')
    
    # Create the boxplot with all features
    plt.figure(figsize=(16, 10))
    ax = sns.boxplot(data=data_melted, x='Feature', y='Value', color='#E1FF58', showfliers=False)

    # Draw the plot to ensure that all elements are created
    plt.draw()

    # Update the color of the median lines
    num_boxes = len(filtered_features)
    lines_per_box = int(len(ax.lines) / num_boxes)

    for i in range(num_boxes):
        # The median line is typically the 5th line in the group, but this can vary
        capline_idx = 3 + i * lines_per_box  # Index of the upper cap line
        median_line_idx = capline_idx + 1  # The median line immediately follows the cap line
        median_line = ax.lines[median_line_idx]
        median_line.set_color('#17354C')
        median_line.set_linewidth(2)

    # Overlay scatter plot on the box plot with customized markers and colors
    for i, feature in enumerate(filtered_features):
        plt.scatter(x=np.random.normal(i, 0.1, size=len(df)), y=df[feature], s=5, alpha=0.1, 
                    color='#643EF0', marker='o', zorder=3)

    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)

    plt.title(f'Outlier Analysis of {dataset_name}', fontsize=14, fontweight='bold', y=1.03)
    plt.xlabel('Features', fontsize=12, labelpad=11)
    plt.ylabel('Numerical Value', fontsize=12, labelpad=11)
    plt.tight_layout(pad=3.0)
    plt.grid(True, zorder=1, color='#DDDDDD', linewidth=0.5) 
    plt.show()


# ---------------------- DATA VISUALIZATION FUNCTIONS ---------------------- #

BT_palette = ["#F72585", "#7209B7", "#3A0CA3", "#4361EE", "#4CC9F0"]
"""
#ddff4f: Light, neon lime green color.
#55109f: Deep purple/blue color.
#e1e057: Pale canary/mustard yellow color.
#3a0ca3: Dark violet color.
"""

# ---------------------- PLOT: ROC & PR-Curve for ONE model ---------------------- #

def plot_roc_and_precision_recall_curves(y_true, y_scores):
    """
    Plot ROC & PR Curves
    =====================

    Plots the Receiver Operating Characteristic (ROC) curve and the Precision-Recall curve with the
    "ideal" and "no skill" model curves for comparison.
    
    Parameters:
    -----------
    - y_true : array-like, shape = [n_samples]
        True binary labels.
    - y_scores : array-like, shape = [n_samples]
        Target scores, probability estimates of the positive class.
        
    Returns:
    ---------
    - roc_auc : float
        Area Under the ROC Curve (AUC) score.
    - pr_auc : float
        Area Under the Precision-Recall Curve (AUC) score.
    """
    # Set up the matplotlib figure and axes
    plt.figure(figsize=(12, 5))

    # Calculate ROC curve and ROC AUC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Precision-Recall curve and PR AUC
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Calculate the "no skill" line as the ratio of positives
    no_skill = sum(y_true) / len(y_true)
    
    # Calculate optimal ROC threshold
    # Find the optimal threshold (Youden’s J statistic)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = roc_thresholds[ix]

    # Plot ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='#FF1F4F', lw=2,label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k-', lw=1, label='Random Model')
    # Plot the "ideal" model curve
    plt.plot([1, 0], [1, 1], color='#5A31EF', lw=2, linestyle='-', label='Perfect Model')
    plt.plot([0, 0], [1, 0], color='#5A31EF', lw=2, linestyle='-')  # Vertical line for recall = 1
    
    # Plot the optimal threshold point
    plt.scatter(fpr[ix], tpr[ix], s=100, color='#CCFF00', edgecolor='black', 
                zorder=5)
    # Adding text next to the dot
    plt.text(fpr[ix] + 0.03, tpr[ix], f'Threshold = {best_thresh:.2f}', fontsize=11, color='#1904DA',
              bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', boxstyle='round,pad=0.1'))

    plt.xlim([-0.05, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=12,labelpad=10)
    plt.ylabel('True Positive Rate',fontsize=12, labelpad=10)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=9)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=9)
    plt.title('ROC Curve (area = %0.2f)' % roc_auc,fontsize=14)
    plt.legend(loc='best', framealpha=1.0,fontsize='small')
    plt.grid(which='major', color='#DDDDDD', linewidth=0.5)

    # Calculate optimal Precision-Recall threshold
    pr_distances = np.sqrt((1-recall)**2 + (1-precision)**2)
    optimal_pr_idx = np.argmin(pr_distances)
    optimal_pr_threshold = pr_thresholds[optimal_pr_idx]
    optimal_pr_precision = precision[optimal_pr_idx]
    optimal_pr_recall = recall[optimal_pr_idx]

    # Plot Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='#FF1F4F', lw=2,label='PR curve')
    plt.plot([0, 1], [no_skill, no_skill], 'k--', lw=1, label='No Skill (AP = %0.2f)' % no_skill)
    # Plot the "ideal" model curve
    plt.plot([0, 1], [1, 1], color='#5A31EF', lw=2, linestyle='-', label='Perfect Model')
    plt.plot([1, 1], [no_skill, 1], color='#5A31EF', lw=2, linestyle='-')  # Vertical line for recall = 1
    # Fill only above the no-skill line
    precision_with_base = np.maximum(precision, no_skill)
    plt.fill_between(recall, precision_with_base, no_skill, alpha=0.3, color='#DFD8F9', step='post')
    # Plot the optimal threshold point
    plt.scatter(optimal_pr_recall, optimal_pr_precision, s=100, color='#CCFF00', edgecolor='black', 
                zorder=5)
    plt.annotate(f'Threshold={optimal_pr_threshold:.2f}', # Text to display
                 (optimal_pr_recall, optimal_pr_precision), # Point to annotate
                 textcoords="offset points", # how to position the text
                 xytext=(7,-1), # distance from text to points (x,y)
                 ha='left', 
                 va='bottom', 
                 color='#1904DA',
                 fontsize=11,
                 bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', boxstyle='round,pad=0.1'))

    plt.xlabel('Recall',fontsize=12, labelpad=10)
    plt.ylabel('Precision',fontsize=12, labelpad=10)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=9)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=9)
    plt.title('PR Curve (area = %0.2f)' % pr_auc,fontsize=14)
    plt.legend(loc='lower center', framealpha=1.0,fontsize='small')
    plt.grid(which='major', color='#DDDDDD', linewidth=0.5)
    # Show the combined plot
    plt.tight_layout()
    plt.show()

    return roc_auc, pr_auc


# -------------------- Confusion Matrix -------------------- #

# Define the colors
start_color = '#ddff4f'  # Neon Yellow
middle_color = '#F72585' # Bright Pink
end_color = '#3A0CA3'   # Assuming typo corrected from '#F3A0CA3' to '#F30CA3'

# Create a custom colormap for the gradient
#n_bins = 100  # adjust for smoother color transitions
#custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", ['#ddff4f','#F72585','#3A0CA3'], N=n_bins)

def plot_confusion_matrix(y_val, y_scores, 
                          threshold=0.5,
                          figsize=(4, 4), 
                          label_fontsize=12, 
                          custom_palette=None):
    """
    Plot Confusion Matrix
    =====================
    Plots the confusion matrix for the given true labels and prediction scores,
    applying a threshold to determine class labels.

    Parameters
    ----------
    y_val : array-like
        True labels.
    y_scores : array-like
        Prediction scores, typically probabilities from a classifier. These are binarized
        to class labels based on the specified threshold.
    threshold : float, optional
        The threshold for binarizing the prediction scores. Defaults to 0.5.
    figsize : tuple, optional
        Figure size for the plot. Default is (4, 4).
    label_fontsize : int, optional
        Font size for the labels in the plot. Default is 12.
    custom_palette : list, optional
        A list of color hex codes for the custom color palette. If None, a default gradient 
        palette is used.

    Returns
    -------
    None
        The function directly plots the confusion matrix and does not return any value.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    >>> model = LogisticRegression()
    >>> model.fit(X_train, y_train)
    >>> y_scores = model.predict_proba(X_test)[:, 1]
    >>> plot_confusion_matrix(y_test, y_scores, threshold=0.5)
    """
    try:
        # Convert scores to binary labels based on the threshold
        y_pred = [1 if score >= threshold else 0 for score in y_scores]

        # Calculate the confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        if cm.shape != (2, 2):
            raise ValueError("Confusion matrix calculation resulted in a shape other than 2x2. Ensure binary classification.")
        
        # Label configuration for the confusion matrix
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        
        # Determine color palette
        if custom_palette:
            if len(custom_palette) != 3:
                logging.warning("Custom palette does not contain exactly 3 colors. Falling back to default palette.")
            else:
                cmap = LinearSegmentedColormap.from_list("Custom", custom_palette, N=100)
        else:
            # Default gradient color palette
            cmap = LinearSegmentedColormap.from_list("custom_gradient", ['#ddff4f', '#F72585', '#3A0CA3'], N=100)
        
        # Plotting
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=labels, fmt='', cmap=cmap, cbar=False, annot_kws={"size": label_fontsize})
        plt.xlabel('Predicted labels', fontsize=9)
        plt.ylabel('Actual labels', fontsize=9)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Confusion Matrix', fontsize=14)
        plt.tight_layout()
        plt.show()

        logging.info("Confusion matrix plotted successfully.")
    except Exception as e:
        logging.error(f"Error plotting confusion matrix: {e}")
        raise


# -------------------- Compare ROC Curves -------------------- #

def compare_roc_curves(y_true, model_preds, model_names):
    """
    Compare ROC Curves
    ==================
    Plots ROC curves for multiple models to compare their performance.

    Args:
    - y_true: Array-like, true binary labels.
    - model_preds: List of arrays containing the predicted probabilities from each model.
    - model_names: List of strings containing model names.
    """
    plt.figure(figsize=(8, 6))
    
    # Define line styles and markers
    line_styles = ['-', '-', '-','-', ':']
    markers = ['o', '^', 'o','s', 'p']
    colors = ['#0A0AFF', '#0FF0FC','#ff0051', '#FF901B', '#3ECEFF', '#FF5E00']
    line_widths = [2,2, 3, 3, 3, 3]  
    
    for i, (preds, name) in enumerate(zip(model_preds, model_names)):
        # Compute ROC curve and ROC area for each model
        fpr, tpr, _ = roc_curve(y_true, preds)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})',
                 linestyle=line_styles[i % len(line_styles)],  # Cycle through line styles
                 color=colors[i % len(colors)],  # Cycle through colors
                 marker=markers[i % len(markers)],  # Cycle through markers
                 linewidth=line_widths[i % len(line_widths)], 
                 markersize=1)
    
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=9)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=9)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Comparison')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.grid(which='major', color='#DDDDDD', linewidth=0.5)
    plt.show()


# -------------------- Compare Results of multiple models visualized -------------------- #

def plot_comparison_chart(comparison_df):
    # Number of groups (models)
    n_groups = len(comparison_df)
    # Create a bar plot for each metric
    fig, ax = plt.subplots(figsize=(8,5))
    index = np.arange(n_groups)
    bar_width = 0.1  # Increase if there is enough space or reduce number of models

    # Specify colors for each metric
    colors = {
        'AUC-ROC': '#ff0051',
        'Precision': '#55109f',
        'Recall': '#ddff4f',
        'F1 Score': '#3a0ca3'
    }
    
    for i, metric in enumerate(['AUC-ROC', 'Precision', 'Recall', 'F1 Score']):
        ax.bar(index + i * bar_width, comparison_df[metric], bar_width, label=metric,
               color=colors[metric])

    # Set labels and titles
    ax.set_xlabel('Model')
    ax.set_ylabel('Value')
    ax.set_title('Comparison of Metrics for Fraud Detection Models')

    # Set x-ticks positions and labels
    ax.set_xticks(index + bar_width / 2)  # Adjust the position of x-ticks to be in the center of the group
    ax.set_xticklabels(comparison_df['Model'])

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

# ---------------------- Function for plotting Individual Feature importances ---------------------- #
def plot_feature_importance(model, feature_names, top_n=None, figsize=(12, 6), title_fontsize=14, label_fontsize=12, color_scheme='viridis'):
    """
    Plots the feature importance for a given model, accommodating various model types.

    This function is designed to be flexible and can extract feature importances from 
    tree-based models (like LightGBM and XGBoost), linear models (like Logistic Regression),
    and potentially other models that provide feature importance measures. It also provides 
    various customization options for the plot.

    Parameters:
    - model (model object): The machine learning model object. Should have a method/attribute 
      for feature importances (e.g., feature_importance(), feature_importances_, coef_).
    - feature_names (list): A list of feature names corresponding to the features in the model.
    - top_n (int, optional): The number of top features to display in the plot. If None, all 
      features are displayed.
    - figsize (tuple): The size of the plot (width, height) in inches.
    - title_fontsize (int): Font size for the plot title.
    - label_fontsize (int): Font size for the labels (x and y axis).
    - color_scheme (str): Color scheme to use for the plot. Can be any valid colormap name in Matplotlib.

    Returns:
    None. Displays a bar plot of the feature importances.

    Raises:
    - AttributeError: If the model does not have a feature importance attribute/method.
    - Exception: For any other issues that arise during execution.
    """
    try:
        # Extracting feature importance based on model type
        if hasattr(model, 'feature_importance'):
            importance = model.feature_importance()
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = model.coef_[0]
        else:
            raise AttributeError("The model does not have a feature importance attribute/method.")

        # Creating DataFrame for feature importance
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        
        # Sort by importance
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

        # If top_n is specified, select top n features
        if top_n is not None:
            feature_importance = feature_importance.head(top_n)

        # Plotting
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette=color_scheme)
        plt.title('Feature Importance', fontsize=title_fontsize)
        plt.xlabel('Importance', fontsize=label_fontsize)
        plt.ylabel('Feature', fontsize=label_fontsize)
        plt.tight_layout()
        plt.show()
        logging.info("Feature importance plotted successfully.")

    except Exception as e:
        logging.error(str(e))


# -------------------- Enhanced SHAP Analysis Plotting Function -------------------- #

def shap_analysis(model, X_val, show_dataframe=False):
    """
    SHAP Analysis
    ============= 
    Performs SHAP analysis on a given model and validation dataset, plots the mean
    absolute SHAP values for each feature, and optionally displays a DataFrame of 
    feature importance based on SHAP values.

    Parameters:
    - model (model object): The trained machine learning model to be analyzed.
    - X_val (DataFrame): The validation dataset used for SHAP analysis.
    - show_dataframe (bool, optional): If True, displays a DataFrame showing features and 
      their corresponding SHAP values in descending order.

    Returns:
    - shap_values (SHAP values object): Calculated SHAP values for the provided dataset.

    Raises:
    - Exception: If an error occurs during SHAP analysis or plotting.
    """
    try:
        # Create object that can calculate SHAP values
        explainer = shap.Explainer(model, X_val)

        # Calculate SHAP values
        shap_values = explainer(X_val, check_additivity=False)

        # Calculate mean absolute SHAP values for each feature
        mean_shap_values = np.abs(shap_values.values).mean(axis=0)

        # Sort the indices of the mean SHAP values in ascending order
        sorted_idx = np.argsort(mean_shap_values)[::-1]

        # Set the figure size
        plt.figure(figsize=(12, 8))  # Adjust the size as needed

        # Create the bar plot
        bars = plt.barh(range(len(sorted_idx)), mean_shap_values[sorted_idx], color='#ff0051')

        # Add text labels to the bars with more decimal places
        for bar in bars:
            plt.text(
                bar.get_width(),  # X position to place the text
                bar.get_y() + bar.get_height() / 2,  # Y position to place the text
                f"{bar.get_width():.3f}",  # Text to be displayed with 3 decimal places
                va='center',  # Vertical alignment
                ha='left'     # Horizontal alignment
            )

        # Set the y-ticks to the feature names
        # Set the y-ticks to the feature names with a specific font size
        plt.yticks(range(len(sorted_idx)), X_val.columns[sorted_idx], fontsize=10)  
        plt.xlabel("Mean Absolute SHAP Value (Impact on Model Output)")
        plt.gca().invert_yaxis()  # Invert y-axis to have the highest value on top
        plt.show()

        # Optionally display a DataFrame of features and SHAP values
        if show_dataframe:
            df_shap = pd.DataFrame({
                'Feature': X_val.columns[sorted_idx],
                'Mean Absolute SHAP Value': mean_shap_values[sorted_idx]
            }).reset_index(drop=True)
            print("\nFeatures and their corresponding SHAP values in descending order:")
            print(df_shap)

        # Log successful completion
        logging.info("SHAP analysis and plotting completed successfully.")

        return shap_values

    except Exception as e:
        logging.error(f"Error during SHAP analysis: {e}")
        raise

# -------------------- Function to display Evaluation metrics -------------------- #

# Function to display DataFrame in a styled format if possible, or plain text otherwise
def display_evaluation_metrics(metrics_dict, title='Evaluation Metrics:', html_style=True):
    df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Score'])
    if html_style:
        try:
            from IPython.display import display
            # Apply the style and set the caption
            styled_df = df.style.format({"Score": "{:.4f}"})\
                                .set_caption(title)\
                                .set_table_attributes('style="margin-left: 0px; width: auto;"')\
                                .set_properties(**{'color': 'black', 'border': '1px solid black', 'text-align': 'left'})\
                                .set_table_styles([
                                    {'selector': 'th, td',
                                     'props': [('padding', '10px'),  # Increase padding
                                               ('font-size', '14pt'),  # Increase font size
                                               ('width', '150px')]},  # Set minimum width for columns
                                    {'selector': 'th', 'props': [('text-align', 'left'), ('width', '100px')]},  # Align headers to the left and set width
                                    {'selector': 'td', 'props': [('text-align', 'left'), ('width', '100px')]},  # Align cell text to the left and set width
                                    {'selector': 'tr:nth-child(even)',
                                     'props': [('background-color', '#f2f2f2')]},  # Light grey background for even rows
                                    {'selector': 'tr:nth-child(odd)',
                                     'props': [('background-color', 'white')]},  # White background for odd rows
                                    {'selector': 'caption',
                                     'props': [('color', 'black'),
                                               ('font-size', '18pt'),  # Increase caption font size
                                               ('font-weight', 'bold'),
                                               ('text-align', 'left')]}
                                ])
            
            # Display the DataFrame
            display(styled_df)
        except ImportError as e:
            print(f"ImportError: {e}")
            print(df.to_string(index=False))
    else:
        # Non-HTML style - for environments that do not render HTML
        print(df.to_string(index=False))

# For HTML-supported environments (e.g., Jupyter Notebooks)
#display_evaluation_metrics(evaluation_results_lgbm)

# For non-HTML environments (e.g., running as a Python script in a terminal)
#display_evaluation_metrics(evaluation_results_lgbm, html_style=False)


# ---------------------- PLOT: PR vs. Threshold Curves ---------------------- #

def plot_pr_tradeoff(y_true, y_scores, figsize=(7, 5)):
    """
    Plots the trade-off between precision and recall
    =================================================
    Plots the trade-off between precision and recall for different thresholds, allowing for the
    identification of an optimal threshold that balances precision and recall.

    Parameters
    ----------
    y_true : array-like
        True binary labels in {0, 1} or {-1, 1}.
    y_scores : array-like
        Target scores, can either be probability estimates of the positive class, confidence values,
        or non-thresholded measure of decisions as returned by a classifier.
    figsize : tuple, optional
        The size of the figure to be created. Default is (7, 5).

    Returns
    -------
    None
        This function does not return a value but shows a matplotlib figure and prints the best threshold
        value where precision and recall are closest.

    Notes
    -----
    This function calculates precision and recall values for each possible threshold and plots them.
    It also finds and marks the threshold where the absolute difference between precision and recall
    is minimized, suggesting a good balance for binary classification tasks.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X, y = make_classification(n_samples=1000, n_features=20, n_informative=2,
    ...                            n_redundant=10, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    >>> model = RandomForestClassifier()
    >>> model.fit(X_train, y_train)
    >>> y_scores = model.predict_proba(X_test)[:, 1]
    >>> plot_precision_recall_tradeoff(y_test, y_scores)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate the differences between precision and recall for each threshold
    differences = np.abs(precision[:-1] - recall[:-1])
    # Find the index of the smallest difference
    min_difference_index = np.argmin(differences)
    # Find the threshold that gives the smallest difference
    best_threshold = thresholds[min_difference_index]
    best_precision = precision[min_difference_index]
    best_recall = recall[min_difference_index]

    # Plotting the precision-recall vs. threshold chart
    plt.figure(figsize=figsize)
    plt.plot(thresholds, precision[:-1], label='Precision', color='#269DAA', linewidth=2)
    plt.plot(thresholds, recall[:-1], '--', label='Recall', color='#230FA0', linewidth=2)
    
    # Highlighting the intersection point
    plt.scatter(best_threshold, best_precision,s=100, color='#ff0051',edgecolor='black', zorder=6)
    
    plt.annotate(f'Threshold={best_threshold:.2f}', # Text to display
                 (best_threshold, best_precision), # Point to annotate
                 textcoords="offset points", # how to position the text
                 xytext=(7,-1), # distance from text to points (x,y)
                 ha='left', 
                 va='bottom', 
                 color='#1904DA',
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', boxstyle='round,pad=0.1'))

    plt.title('Precision-Recall vs. Threshold', fontsize=14)
    plt.xlabel('Threshold', fontsize=12,labelpad=10)
    plt.ylabel('Score', fontsize=12, labelpad=10)
    plt.xticks(np.arange(0, 1.05, 0.1), fontsize=9)
    plt.yticks(np.arange(0, 1.05, 0.1), fontsize=9)
    plt.legend(loc='best', framealpha=1.0,fontsize='small')
    plt.grid(which='major', color='#DDDDDD', linewidth=0.5)
    plt.show()
    print(f"Best Threshold for equal Precision and Recall: {best_threshold:.2f}")



# ---------------------- PLOT: ADV Learning Curves ---------------------- #

def plot_advanced_learning_curves(train_metrics_df, 
                                  val_metrics_df, 
                                  metrics, 
                                  y_limits=None):
    """
    Plots advanced learning curves
    ================================
    Plots advanced learning curves with error bars for specified evaluation metrics.
    The function plots training and validation scores for each metric, along with the
    standard deviation as shaded areas around the lines, allowing for visual assessment
    of the model's performance and its variance.

    Parameters:
    - train_metrics_df (pd.DataFrame): DataFrame containing the training metrics for each fold.
    - val_metrics_df (pd.DataFrame): DataFrame containing the validation metrics for each fold.
    - metrics (list): List of metrics to be plotted.
    - y_limits (tuple, optional): Tuple containing y-axis limits (min, max). If not provided,
                                  the function calculates them based on the metric values.

    Returns:
    - None: The function does not return anything but plots the learning curves.
    """

    # Set the number of plots based on the number of metrics
    num_metrics = len(metrics)
    cols = 2  # Define the number of columns for subplots
    rows = num_metrics // cols + bool(num_metrics % cols)  # Calculate rows needed based on the number of metrics
    
    # Set the plot size and layout
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    
    # Ensure axs is a flat list for consistent indexing
    axs = axs.flatten() if num_metrics > 1 else [axs]
    
    # Calculate common y-axis limits if not provided
    if y_limits is None:
        all_values = np.concatenate([train_metrics_df[m].values for m in metrics] + [val_metrics_df[m].values for m in metrics])
        y_min, y_max = np.min(all_values), np.max(all_values)
        y_limits = (y_min - 0.05 * np.abs(y_min), y_max + 0.05 * np.abs(y_max))  # Add 5% padding for visibility

    # Loop through each metric and create its subplot
    for i, metric in enumerate(metrics):
        ax = axs[i] if num_metrics > 1 else axs[0]
        
        # Prepare mean and standard deviation for training and validation metrics
        train_means = train_metrics_df[metric].values
        val_means = val_metrics_df[metric].values
        train_std = train_metrics_df[metric].std()
        val_std = val_metrics_df[metric].std()
        
        x_axis = np.arange(1, len(train_means) + 1)  # x-axis values (number of folds)

        # Plot learning curves with shaded error bars
        ax.fill_between(x_axis, train_means - train_std, train_means + train_std, alpha=0.2, color='#5A31EF')
        ax.fill_between(x_axis, val_means - val_std, val_means + val_std, alpha=0.2, color="#FF901B")
        ax.plot(x_axis, train_means, 'o-', color='#5A31EF', label="Training scores", linewidth=1)
        ax.plot(x_axis, val_means, 's-', color="#FF0051", label="Cross-validation scores", linewidth=1)

        # Set individual plot titles, labels, and y-axis limits
        ax.set_title(f'{metric} - Learning Curve', fontsize=14)
        ax.set_xlabel('Number of Folds\n', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.tick_params(axis='y', labelsize=10)  # Change '10' to your desired font size
        ax.tick_params(axis='x', labelsize=10)
        ax.set_ylim(y_limits)
        ax.set_xticks(x_axis) 
        ax.legend(loc='best', framealpha=1.0, fontsize='small')
        ax.grid(which='major', color='#DDDDDD', linewidth=0.5)

    # Hide any unused subplot areas if the number of metrics is not a perfect fit for the grid layout
    if num_metrics % cols:
        for idx in range(num_metrics, rows * cols):
            axs[idx].set_visible(False)

    plt.suptitle('Advanced Learning Curves', fontsize=16)  # Set the main title for the plot

    plt.show()  # Display the plot

# ---------------------- PLOT: INTERACTIVE Learning Curves ---------------------- #

def plot_interactive_learning_curves(train_metrics_df, 
                                     val_metrics_df, 
                                     metrics, 
                                     y_limits=None):
    """
    Plot advanced plotly learning curves
    ====================================

    Plots interactive learning curves with error bars for specified evaluation metrics.
    The function plots training and validation scores for each metric, along with the
    standard deviation as shaded areas around the lines, allowing for visual assessment
    of the model's performance and its variance.

    Parameters:
    -----------

    - train_metrics_df (pd.DataFrame): DataFrame containing the training metrics for each fold.
    - val_metrics_df (pd.DataFrame): DataFrame containing the validation metrics for each fold.
    - metrics (list): List of metrics to be plotted.
    - y_limits (tuple, optional): Tuple containing y-axis limits (min, max). If not provided,
                                  the function calculates them based on the metric values.

    Returns:
    --------

    - None: The function does not return anything but plots the learning curves.
    """
    # Initialize Plotly for offline use
    init_notebook_mode(connected=True)

    # Define valid metrics
    valid_metrics = {'precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'balanced_acc'}
    
    # Filter out invalid metrics and log a message
    filtered_metrics = [metric for metric in metrics if metric in valid_metrics]
    invalid_metrics = set(metrics) - valid_metrics
    if invalid_metrics:
        logging.info(f"Skipping invalid metrics: {', '.join(invalid_metrics)}")

    # Set the number of plots based on the number of metrics
    num_metrics = len(metrics)
    cols = 2  # Define the number of columns for subplots
    rows = num_metrics // cols + bool(num_metrics % cols)  # Calculate rows needed based on the number of metrics

    # Create subplots
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'{metric} - Learning Curve' for metric in metrics])

    # Calculate common y-axis limits if not provided
    if y_limits is None:
        all_values = np.concatenate([train_metrics_df[m].values for m in metrics] + [val_metrics_df[m].values for m in metrics])
        y_min, y_max = np.min(all_values), np.max(all_values)
        y_limits = (y_min - 0.05 * np.abs(y_min), y_max + 0.05 * np.abs(y_max))  # Add 5% padding for visibility

    # Loop through each metric and create its subplot
    for i, metric in enumerate(metrics):
        row = i // cols + 1
        col = i % cols + 1
        
        # Prepare mean and standard deviation for training and validation metrics
        train_means = train_metrics_df[metric].values
        val_means = val_metrics_df[metric].values
        train_std = train_metrics_df[metric].std()
        val_std = val_metrics_df[metric].std()
        
        x_axis = np.arange(1, len(train_means) + 1)  # x-axis values (number of folds)
        
        # Add training curves with shaded error bars
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=train_means,
            mode='lines+markers',
            name='Training',
            line=dict(color='#5A31EF'),
            error_y=dict(
                type='data',
                array=train_std * np.ones_like(train_means),
                visible=True,
                color='#5A31EF',
                thickness=1.5
            ),
            showlegend=(i == 0),  # Only show legend once for Training
            legendgroup='Training',
            hovertemplate='Fold Number: %{x}<br>Score: %{y:.4f}<extra></extra>',
            hoverlabel=dict(font_size=18)  # Set hover text font size
        ), row=row, col=col)

        # Add validation curves with shaded error bars
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=val_means,
            mode='lines+markers',
            name='Validation',
            line=dict(color='#FF0051'),
            error_y=dict(
                type='data',
                array=val_std * np.ones_like(val_means),
                visible=True,
                color='#FF0051',
                thickness=1.5
            ),
            showlegend=(i == 0),  # Only show legend once for Validation
            legendgroup='Validation',
            hovertemplate='Fold Number: %{x}<br>Score: %{y:.4f}<extra></extra>',
            hoverlabel=dict(font_size=18)  # Set hover text font size
        ), row=row, col=col)

        # Update individual subplot layout
        fig.update_xaxes(
            title_text='Number of Folds',
            row=row, col=col,
            showgrid=True, 
            gridcolor='#DDDDDD',  # Light gray grid lines
            gridwidth=1  # Set grid line width
        )
        fig.update_yaxes(
            title_text='Score',
            range=y_limits,
            row=row, col=col,
            showgrid=True, 
            gridcolor='#DDDDDD',  # Light gray grid lines
            gridwidth=1  # Set grid line width
        )

    # Update overall layout
    fig.update_layout(
        title=dict(
            text='Interactive Learning Curves',
            font=dict(size=25)  # Set the title font size
        ),
        legend=dict(
            x=0.5, y=1.06, orientation='h', xanchor='center', yanchor='top',
            itemclick='toggleothers',
            font=dict(size=18)  # Set the legend font size
        ),
        margin=dict(l=40, r=40, t=100, b=40),  # Adjust top margin for bigger title
        height=rows * 600,  # Adjust height for the number of rows
        width=cols * 900,  # Adjust width for the number of columns
        plot_bgcolor='#FFFFFF',  # Light background color for the plot area
        paper_bgcolor='#FFFFFF'  # Light background color for the entire figure
    )

    #fig.show()

    # Use iplot to display the figure inline
    iplot(fig)


# ---------------------- PLOT: Cumulative gain and lift charts for one model ---------------------- #
def find_elbow_point(percentages, gains):
    # Find the point on the curve furthest away from the line between the curve's start and end
    start = np.array([percentages[0], gains[0]])
    end = np.array([percentages[-1], gains[-1]])
    line_vec = end - start
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    distances = []
    for p, g in zip(percentages, gains):
        vec = np.array([p, g]) - start
        line_proj = start + np.dot(vec, line_vec_norm) * line_vec_norm
        distances.append(np.sqrt(np.sum((vec - line_proj)**2)))
    elbow_index = np.argmax(distances)
    return percentages[elbow_index], gains[elbow_index]


def plot_cumulative_gain(y_true, y_score, threshold=0.5, pos_label=1):
    """
    Plot cumulative gain chart
    ===========================

    Plots the cumulative gain chart for the provided true labels and scores.

    Parameters:
    -----------

    - y_true (array-like): True binary labels.
    - y_score (array-like): Target scores, can either be probability estimates of the positive class.
    - pos_label (int, optional): Label of the positive class.
    """
    # Calculate the cumulative gains
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]
    gains = np.cumsum(y_true_sorted == pos_label) / np.sum(y_true == pos_label)
    percentages = np.linspace(0, 1, len(gains) + 1)  # Create a percentages array that matches the gains array length

    # Calculate the lift values
    lift_values = gains / percentages[1:]  # Exclude the first element which is 0 to avoid division by 0

    # Calculate the elbow point for the trade-off
    elbow_percentage, elbow_gain = find_elbow_point(percentages[1:], gains)

    # Calculate the perfect model gains
    sorted_true_labels = np.sort(y_true)[::-1]  # Sort true labels in descending order
    perfect_gains = np.cumsum(sorted_true_labels == pos_label) / np.sum(y_true == pos_label)
    perfect_gains = np.insert(perfect_gains, 0, 0)  # Add a 0 at the beginning for the perfect model

    # Determine the index of the threshold value
    threshold_index = next(x[0] for x in enumerate(sorted_indices) if y_score[x[1]] < threshold)
    # Get the percentage of samples and gain at the threshold
    threshold_percentage = percentages[threshold_index]
    threshold_gain = gains[threshold_index]

    # Number of positives in y_true
    num_positives = np.sum(y_true == pos_label)
    
    # Sort y_score in descending order and get the top 30%
    sorted_score_indices = np.argsort(y_score)[::-1]
    top_30_percent_index = int(0.3 * len(y_score))
    # Find the cumulative gain at the top 30%
    gains_at_top_30 = np.cumsum(y_true[sorted_score_indices][:top_30_percent_index] == pos_label) / num_positives
    gain_at_3rd_decile = gains_at_top_30[-1]  # The last value in gains_at_top_30 is the cumulative gain at the 3rd decile
    # Find the percentage value at the 3rd decile for plotting
    percent_3rd_decile = 0.3  # The 3rd decile is at 30% of the total sample

    # Plot the charts
    plt.figure(figsize=(12, 6))

# ---------------------- PLOT: Cumulative Gains Chart
    plt.subplot(1, 2, 1)
    plt.plot(percentages, perfect_gains, '#5A31EF', label='Perfect Model', linewidth=2)
    plt.plot(percentages[1:], gains, label='Actual Model (Class 1)', color='#FF1F4F', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k-', label='Random Model', linewidth=1)
    
    # Add the threshold dot
    plt.scatter(threshold_percentage, threshold_gain, s=100, c='#BAEEFF', edgecolor='black', zorder=5, label=f'Threshold at {threshold}')
    plt.annotate(f'{threshold_gain*100:.2f}%', # Text to display
                (threshold_percentage, threshold_gain), # Point to annotate
                textcoords="offset points", # how to position the text
                xytext=(13,-4), # distance from text to points (x,y)
                ha='left', 
                fontsize=11,
                bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', boxstyle='round,pad=0.1')
                )
    
    # Plot the elbow point as a trade-off point
    plt.scatter(elbow_percentage, elbow_gain, s=100, c='#D6DAFF', edgecolor='black', zorder=5, label='Trade-off Point')

    # Plot a vertical line at the 3rd decile (30% of the sample)
    plt.axvline(x=percent_3rd_decile, color='grey', linestyle='--', linewidth=2)
    # Mark the intersection point
    plt.scatter(percent_3rd_decile, gain_at_3rd_decile, s=100, c='#E0FF58',edgecolor='black', zorder=5, label='Top 30% Gain')
    plt.annotate(f'{gain_at_3rd_decile*100:.2f}%', # Text to display
                 (percent_3rd_decile, gain_at_3rd_decile), # Point to annotate
                 textcoords="offset points", # how to position the text
                 xytext=(10,-4), # distance from text to points (x,y)
                 ha='left', fontsize=11,
                 bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', boxstyle='round,pad=0.1')
                 )

    plt.title('Cumulative Gains Chart', fontsize=14)
    plt.xlabel('Perc. of total sample tested', fontsize=12, labelpad=10)
    plt.ylabel('Perc. of positive samples found',fontsize=12, labelpad=10)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=9)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=9)
    plt.legend(loc='best', framealpha=1.0,fontsize='small')
    plt.grid(which='major', color='#DDDDDD', linewidth=0.5)

    plt.tight_layout()
    plt.show()

# ---------------------- COMPARISON PLOT: Cumulative gain and lift charts for MULTIPLE models ---------------------- #

def calculate_lift(y_true, y_score):
    # Sort by score
    sorted_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Calculate cumulative sum of true positives
    cumulative_positives = np.cumsum(y_true_sorted)
    
    # Calculate lift
    lift = cumulative_positives / (np.arange(1, len(y_true) + 1) * np.mean(y_true_sorted))
    return lift


def plot_multiple_cumulative_gains(models_dict, threshold=0.5, pos_label=1):
    """
    Plot multiple cumulative gains charts
    ======================================

    Plot cumulative gains and lift charts for multiple models on the same axes for comparison.

    Parameters
    ----------

    models_dict : dict
        A dictionary where keys are model names (str) and values are tuples containing 
        actual labels and predicted scores (numpy arrays or lists).
    threshold : float, optional
        The decision threshold for classifying positive vs negative. Default is 0.5.
    pos_label : int, optional
        The label of the positive class. Default is 1.

    Notes
    -----

    This function creates two subplots: the left subplot displays the cumulative gains 
    chart, and the right subplot (placeholder in this function) would typically display the 
    lift chart. Each model's performance is plotted using a unique color.

    Examples
    --------
    >>> models_dict = {
        "Model A": (y_true_a, y_scores_a),
        "Model B": (y_true_b, y_scores_b)
    }
    >>> plot_multiple_cumulative_gains(models_dict)

    """
    plt.figure(figsize=(14, 7))

    # Define color cycle
    colors = cycle(['#0A0AFF', '#CE65FB','#ff0051', '#FF901B', '#2CC2D3', '#FF11FF'])

    # Cumulative Gains Chart
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], 'k-', linewidth=1)

    third_decile_line = 0.3

    # Plot the 3rd decile line (30% vertical line)
    plt.axvline(x=third_decile_line, color='grey', linestyle='--', linewidth=2)

    # Lift Chart setup (will be filled in the loop)
    lift_chart_data = []

    # Store model names to ensure color consistency
    model_color_map = {}

    # Store gain values for legend labels
    gain_values_for_legend = {}

    # Loop over each model
    for model_name, (y_true, y_score) in models_dict.items():
        color = next(colors)
        model_color_map[model_name] = color  # Map the model to the color

        y_true = np.array(y_true)
        y_score = np.array(y_score)
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]
        gains = np.cumsum(y_true_sorted == pos_label) / np.sum(y_true == pos_label)

        gains = np.insert(gains, 0, 0)  # Insert a 0 at the beginning for 0% tested

        # Define percentages with the correct length
        percentages = np.linspace(0, 1, len(gains))

        # Calculate lift values for the Lift Chart
        lift_values = gains/ percentages  # Skip the first percentage to avoid division by zero

        lift_chart_data.append((percentages[1:], lift_values, model_name))

        # Calculate and plot the elbow point for each model - Trade Off
        elbow_percentage, elbow_gain = find_elbow_point(percentages[1:], gains)
        plt.plot(percentages, gains, label=f'{model_name}', linewidth=1.5, color=color)
        plt.scatter(elbow_percentage, elbow_gain, s=40, c=color, edgecolor='None', zorder=4)

        # Calculate and plot the top 30% gain for each model
        top_30_percent_index = int(third_decile_line * len(gains))
        gain_at_top_30_percent = gains[top_30_percent_index]
        plt.scatter(third_decile_line, gain_at_top_30_percent, s=50, 
                    color='#E0FF58', edgecolor=color, zorder=5)

        # plt.annotate(f'{gain_at_top_30_percent*100:.2f}%', # Text to display
        #             (third_decile_line, gain_at_top_30_percent), # Point to annotate
        #             textcoords="offset points", # how to position the text
        #             xytext=(20,0), # distance from text to points (x,y)
        #             ha='left', fontsize=8, color=color,
        #             bbox=dict(facecolor='white', edgecolor='none', pad=0.5)
        #             )
        # Store the gain for legend labels
        gain_values_for_legend[model_name] = gain_at_top_30_percent*100

    # Custom legend entries for models including the top 30% gain
    model_legend_entries = [
        mlines.Line2D([], [],  marker='o', markerfacecolor='#E0FF58', markeredgecolor='black', markersize=7, 
                    color=model_color_map[model_name], linestyle='-', linewidth=2,
                    label=f'{model_name} ({gain_values_for_legend[model_name]:.2f}%)') 
        for model_name in models_dict.keys()
    ]
    
    # Custom legend entries for points
    trade_off_dot = mlines.Line2D([], [], color='black', marker='o', linestyle='None', 
                                  markersize=7, label='Trade-off', markeredgecolor='black')
    
    # Combine model legend entries with custom point entries
    handles = model_legend_entries + [trade_off_dot]
    
    # Plot Perfect Model (only once)
    sorted_true_labels = np.sort(y_true)[::-1]
    perfect_gains = np.cumsum(sorted_true_labels == pos_label) / np.sum(y_true == pos_label)
    perfect_gains = np.insert(perfect_gains, 0, 0)
    plt.plot(percentages, perfect_gains, '#5A31EF', linewidth=2)
    plt.title('Cumulative Gains Chart', fontsize=16)
    plt.xlabel('Perc. of total sample tested', fontsize=12, labelpad=10)
    plt.ylabel('Perc. of positive samples found', fontsize=12, labelpad=10)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=9)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=9)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.5)
    plt.tight_layout()

    # Create the legend with a title
    plt.legend(handles=handles, loc='best', framealpha=1.0, fontsize=11, 
               title='Top 30% Gain', title_fontsize=13)
    plt.show()


# ---------------------- COMPARISON PLOT: ROC & PR-AUC Curves for MULTIPLE Models ---------------------- #

def plot_multiple_roc_prc(models_data):
    """
    Plot multiple ROC and PR Curves
    ================================

    Plots the Receiver Operating Characteristic (ROC) and Precision-Recall curves for multiple models.

    Parameters:
    -----------

    - models_data: dict
        Dictionary containing model names as keys and tuples (y_true, y_scores) as values.

    """
    # Define color cycle
    colors = cycle(['#0A0AFF', '#CE65FB','#ff0051', '#FF901B', '#2CC2D3', '#FF11FF'])

    # Set up the matplotlib figure and axes
    plt.figure(figsize=(12, 6))

    # Store model names to ensure color consistency
    model_color_map = {}

    # Store threshold values
    best_thresh_legend = {}
    auc_legend = {}
    pr_legend = {}
    optimal_pr_threshold_legend= {}

    # Plot ROC and Precision-Recall for each model
    for model_name, (y_true, y_scores) in models_data.items():
        color = next(colors)
        model_color_map[model_name] = color  # Map the model to the color

        # ---------------------- ROC Curve ---------------------- #

        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        # Calculate optimal ROC threshold # Find the optimal threshold (Youden’s J statistic)
        J = tpr - fpr
        ix = np.argmax(J)
        best_thresh = roc_thresholds[ix]

        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{model_name}: area = {roc_auc:.2f}; ')

        # Calculate the point on the ROC curve for the ordinary threshold at 0.5
        ordinary_threshold = 0.5
        fpr_ordinary, tpr_ordinary, _ = roc_curve(y_true, y_scores)
        # Get closest threshold index to 0.5 (this may not be exactly 0.5)
        ix_ordinary = np.argmin(np.abs(roc_thresholds - ordinary_threshold))

        # Plot ORDINARY Thresh.
        plt.scatter(fpr_ordinary[ix_ordinary], tpr_ordinary[ix_ordinary],marker='D', s=40, 
                    color=model_color_map[model_name], edgecolor=color, 
                    zorder=4)

        # Plot OPTIMAL Thresh.
        plt.scatter(fpr[ix], tpr[ix], s=70, color='#CCFF00', edgecolor=model_color_map[model_name], 
                    zorder=5)
        # Adding text next to the dot
        # plt.text(fpr[ix] + 0.03, tpr[ix], f'Threshold = {best_thresh:.2f}', fontsize=9, color=color,
        #         bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', boxstyle='round,pad=0.1'))
        
        # Store the gain and AUC for legend labels
        best_thresh_legend[model_name] = f'{best_thresh:.2f}'
        auc_legend[model_name] = f'{roc_auc:.3f}'

        # ---------------------- PR Curve ---------------------- #
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        # Calculate the F1 score for each point on the PR curve
        f1_scores = 2 * (precision * recall) / (precision + recall)
        # Find the index of the maximum F1 score
        ix_f1_max = np.argmax(f1_scores)

        # This is the optimal threshold based on F1 score
        optimal_pr_threshold = pr_thresholds[ix_f1_max]
        optimal_precision = precision[ix_f1_max]
        optimal_recall = recall[ix_f1_max]
        pr_auc = auc(recall, precision)

        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color=color, lw=2, label=f'{model_name} (AUC = {pr_auc:.2f})')
        
        # Plot the optimal threshold point
        plt.scatter(optimal_recall, optimal_precision, marker='o', color='#CCFF00', edgecolor=color, s=70, zorder=4)
        
        # Store the gain and AUC for legend labels
        optimal_pr_threshold_legend[model_name] = f'{optimal_pr_threshold:.2f}'
        pr_legend[model_name] = f'{pr_auc:.3f}'

# ---------------------- ROC Curve ---------------------- #
    # ROC: Custom legend entries for models including best threshold
    model_legend_entries = [
        mlines.Line2D([], [],  marker='o', markerfacecolor='#CCFF00', markeredgecolor='black', markersize=7, 
                    color=model_color_map[model_name], linestyle='-', linewidth=2,
                    label=f'{model_name}: {auc_legend[model_name]}  ({best_thresh_legend[model_name]})') 
        for model_name, color in zip(models_data.keys(), colors)
    ]
    
    # Custom legend entries for points
    ordinary_thresh_dot = mlines.Line2D([], [], marker='D', markersize=6, markerfacecolor='black',
                                        linestyle='None', label='Ordinary thresh. (0.5)', markeredgecolor='None')
    
    # Combine model legend entries with custom point entries
    handles = model_legend_entries + [ordinary_thresh_dot]

    # ROC subplot settings
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Model')
    # Plot the "ideal" model curve
    plt.plot([1, 0], [1, 1], color='#5A31EF', lw=2, linestyle='-', label='Perfect Model')
    plt.plot([0, 0], [1, 0], color='#5A31EF', lw=2, linestyle='-')  # Vertical line for recall = 1
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, labelpad=10)
    plt.ylabel('True Positive Rate', fontsize=12, labelpad=10)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=9)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=9)
    plt.title('ROC Curves', fontsize=16)
    plt.legend(handles=handles, loc='lower right', 
               fontsize='small', title_fontsize=13,framealpha=1.0,
                title='Area & best Thresh')

    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)

# ---------------------- PR Curve ---------------------- #
    # PR: Custom legend entries for models including best threshold
    model_legend_entries_prc = [
        mlines.Line2D([], [],  marker='o', markerfacecolor='#CCFF00', markeredgecolor='black', markersize=7, 
                    color=model_color_map[model_name], linestyle='-', linewidth=2,
                    label=f'{model_name}: {pr_legend[model_name]}  ({optimal_pr_threshold_legend[model_name]})') 
        for model_name, color in zip(models_data.keys(), colors)
    ]
    
    # Combine model legend entries with custom point entries
    handles_prc = model_legend_entries_prc 
    
    # PR subplot settings
    plt.subplot(1, 2, 2)
     # No Skill Line: Assuming a constant classifier that predicts positives for all instances
    no_skill = np.sum(y_true) / len(y_true)  # Proportion of positives
    plt.plot([0, 1], [no_skill, no_skill], 'k--', lw=1, label='No Skill')

    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, labelpad=10)
    plt.ylabel('Precision', fontsize=12, labelpad=10)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=9)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=9)
    plt.title('Precision-Recall Curves', fontsize=16)

    plt.legend(handles=handles_prc, loc='lower left', 
            fontsize='small', title_fontsize=13,framealpha=1.0,
            title='Area & best Thresh')
    
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)

    # Show the combined plot
    plt.tight_layout()
    plt.show()


# ---------------------- COMPARISON PLOT: Confusion Matrices for MULTIPLE Models ---------------------- #

def panel_of_confusion_matrices(models_data, threshold=0.5,                             
                                custom_palette=None):
    """
    Plot multiple confusion matrices
    =================================

    Plots multiple confusion matrices for the given dictionary of model data in a grid layout.
    
    Parameters:
    -----------

    - models_data (dict): Dictionary where keys are model names and values are tuples of (y_val, y_pred_prob).
    - threshold (float): Classification threshold to convert probabilities to binary class labels.
    - custom_palette (list): A list of color hex codes for the custom color palette. If None, a default gradient palette is used.

    Returns:
    --------

    - None: The function directly plots the confusion matrices.
    """
    figsize_per_cm=(3,3)
    label_fontsize=12
    title_fontsize=13

    num_models = len(models_data)
    grid_size = math.ceil(math.sqrt(num_models))
    fig_width = figsize_per_cm[0] * grid_size
    fig_height = figsize_per_cm[1] * grid_size
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(fig_width, fig_height))
    
    axes = axes.flatten()
    for i in range(num_models, len(axes)):
        axes[i].axis('off')
        
    if custom_palette and len(custom_palette) == 3:
        cmap = LinearSegmentedColormap.from_list("Custom", custom_palette, N=100)
    else:
        cmap = LinearSegmentedColormap.from_list("custom_gradient", ['#ddff4f', '#F72585', '#3A0CA3'], N=100)
        
    for ax, (model_name, (y_val, y_pred_prob)) in zip(axes, models_data.items()):
        # Convert probabilities to binary class labels
        y_pred = [1 if prob >= threshold else 0 for prob in y_pred_prob]
        cm = confusion_matrix(y_val, y_pred)
        group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        sns.heatmap(cm, annot=labels, fmt='', cmap=cmap, cbar=False, annot_kws={"size": label_fontsize}, ax=ax)
        ax.set_xlabel('Predicted labels', fontsize=10)
        ax.set_ylabel('Actual labels', fontsize=10)
        ax.set_title(f'{model_name} Confusion Matrix', fontsize=title_fontsize)
        ax.set_xticks(np.arange(2) + 0.5)
        ax.set_xticklabels(['Negative', 'Positive'], fontsize=9)
        ax.set_yticks(np.arange(2) + 0.5)
        ax.set_yticklabels(['Negative', 'Positive'], fontsize=9)
    
    plt.tight_layout()
    plt.show()
    logging.info("Multiple confusion matrices plotted successfully.")


# -------------------- EVALUATION: Compare Train & Test metrics -------------------- #

def compare_train_val_metrics(metrics_input, model_name='Model', figsize=(8, 5)):
    """
    Plot training and validation metrics
    ======================================

    Compare and plot training and validation metrics for a given model.
    This function accepts either a DataFrame with specific columns or two dictionaries.

    Parameters:
    -----------

    metrics_input (DataFrame or tuple of dicts): Either a DataFrame containing the metrics
                                                  or a tuple containing two dictionaries for
                                                  training and validation metrics, respectively.
    model_name (str): Name of the model being evaluated.
    figsize (tuple): Size of the figure to be plotted.
    
    Returns:
    --------

    None: This function plots the metrics comparison.
    """
    try:
        # Check if input is DataFrame and convert to dictionaries
        if isinstance(metrics_input, pd.DataFrame):
            # Convert the DataFrame into two separate dictionaries for training and validation
            train_metrics = metrics_input.set_index('Metric')['Train Set'].to_dict()
            val_metrics = metrics_input.set_index('Metric')['Test Set'].to_dict()
        elif isinstance(metrics_input, tuple) and len(metrics_input) == 2:
            train_metrics, val_metrics = metrics_input
        else:
            raise ValueError("Input must be either a DataFrame or a tuple of two dictionaries.")

        # Extract metrics scores, ensuring they are present in both dictionaries
        metrics_names = list(train_metrics.keys())
        train_scores = [train_metrics.get(name, 0) for name in metrics_names]
        val_scores = [val_metrics.get(name, 0) for name in metrics_names]

        # Plotting
        n_groups = len(metrics_names)
        index = np.arange(n_groups)
        bar_width = 0.3

        fig, ax = plt.subplots(figsize=figsize)
        bars1 = ax.bar(index, train_scores, bar_width, label='Train', color='#472081', zorder=5)
        bars2 = ax.bar(index + bar_width, val_scores, bar_width, label='Test', color='#7885FF',zorder=5)

        # Add labels above the bars
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            fontsize=10,
                            ha='center', va='bottom')

        add_labels(bars1)
        add_labels(bars2)

        # Add a horizontal red line at y=0.5
        ax.axhline(y=0.5, color='#FF2F5C', linestyle='--', linewidth=1.5, zorder=5)

        ax.set_xlabel('\nMetrics',fontsize=10)
        ax.set_ylabel('Scores',fontsize=10)
        ax.set_title(f'\nTrain vs. Test Metrics for {model_name}',fontsize=14)
        ax.set_xticks(index + bar_width / 3)
        ax.set_xticklabels(metrics_names,fontsize=10)

        ax.legend(framealpha=1.0, loc="lower center",fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.tick_params(axis='y', labelsize=10)

        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.grid(color='#DDDDDD', linewidth=0.6)
        plt.show()

    except Exception as e:
        logging.error(f"Error occurred while comparing metrics: {e}")