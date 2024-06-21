import pandas as pd
import dask.dataframe as dd
import logging
import matplotlib.pyplot as plt
import json

try:
    import cudf
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataQualityChecker:
    def __init__(self, df, npartitions=4, threshold=1.5):
        self.npartitions = npartitions
        self.threshold = threshold
        self.df = self._initialize_df(df)
        self.is_gpu = HAS_GPU

    def _initialize_df(self, df):
        if HAS_GPU:
            logging.info("Initializing with cuDF for GPU acceleration.")
            if isinstance(df, pd.DataFrame):
                return cudf.DataFrame.from_pandas(df)
            elif isinstance(df, cudf.DataFrame):
                return df
            else:
                raise ValueError("Input should be a pandas or cuDF DataFrame")
        else:
            logging.info("Initializing with Dask for parallel processing.")
            if isinstance(df, pd.DataFrame):
                return dd.from_pandas(df, npartitions=self.npartitions)
            elif isinstance(df, dd.DataFrame):
                return df
            else:
                raise ValueError("Input should be a pandas or dask DataFrame")

    def _print_and_return(self, result):
        print(result)
        return result

    def check_missing_values(self):
        logging.info("Checking for missing values...")
        try:
            if self.is_gpu:
                missing_values = self.df.isnull().sum()
                missing_percentage = (missing_values / len(self.df)) * 100
                result = cudf.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage}).to_pandas()
            else:
                missing_values = self.df.isnull().sum().compute()
                missing_percentage = (missing_values / len(self.df)) * 100
                result = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
            return self._print_and_return(result)
        except Exception as e:
            logging.error(f"Error checking missing values: {e}")
            raise

    def check_duplicates(self):
        logging.info("Checking for duplicate rows...")
        try:
            if self.is_gpu:
                result = self.df[self.df.duplicated()].to_pandas()
            else:
                duplicated_mask = self.df.map_partitions(lambda df: df.duplicated(keep=False))
                result = self.df[duplicated_mask].compute()
            return self._print_and_return(result)
        except Exception as e:
            logging.error(f"Error checking duplicates: {e}")
            raise

    def check_data_types(self):
        logging.info("Checking data types...")
        try:
            data_types = self.df.dtypes
            result = pd.Series(data_types, name='Data Types') if self.is_gpu else data_types
            return self._print_and_return(result)
        except Exception as e:
            logging.error(f"Error checking data types: {e}")
            raise

    def check_outliers(self):
        logging.info("Checking for outliers...")
        try:
            numeric_cols = self.df.select_dtypes(include=['float', 'int'])
            outlier_report = {}
            for col in numeric_cols.columns:
                if self.is_gpu:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                else:
                    Q1 = self.df[col].quantile(0.25).compute()
                    Q3 = self.df[col].quantile(0.75).compute()
                IQR = Q3 - Q1
                lower_bound = Q1 - (self.threshold * IQR)
                upper_bound = Q3 + (self.threshold * IQR)
                if self.is_gpu:
                    outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                    outlier_report[col] = outliers.to_pandas()
                else:
                    outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].compute()
                    outlier_report[col] = outliers
                logging.info(f"Outliers in column {col}:")
                logging.info(outlier_report[col])
            return outlier_report
        except Exception as e:
            logging.error(f"Error checking outliers: {e}")
            raise

    def check_value_ranges(self, column_ranges):
        logging.info("Checking for value ranges...")
        try:
            range_report = {}
            for col, (min_val, max_val) in column_ranges.items():
                if self.is_gpu:
                    out_of_range = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)]
                    range_report[col] = out_of_range.to_pandas()
                else:
                    out_of_range = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)].compute()
                    range_report[col] = out_of_range
                logging.info(f"Values out of range in column {col}:")
                logging.info(range_report[col])
            return range_report
        except Exception as e:
            logging.error(f"Error checking value ranges: {e}")
            raise

    def run_all_checks(self, column_ranges=None):
        logging.info("Running all data quality checks...")
        try:
            checks = {}
            checks['Missing Values'] = self.check_missing_values()
            checks['Duplicates'] = self.check_duplicates()
            checks['Data Types'] = self.check_data_types()
            checks['Outliers'] = self.check_outliers()
            if column_ranges:
                checks['Range Validation'] = self.check_value_ranges(column_ranges)
            logging.info("All checks completed.")
            return checks
        except Exception as e:
            logging.error(f"Error running all checks: {e}")
            raise

    def visualize_outliers(self, outlier_report):
        logging.info("Visualizing outliers...")
        try:
            for col, outliers in outlier_report.items():
                if not outliers.empty:
                    plt.figure(figsize=(10, 6))
                    plt.hist(outliers[col], bins=20, edgecolor='k', alpha=0.7)
                    plt.title(f'Outliers in {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.grid(True)
                    plt.show()
                else:
                    logging.info(f"No outliers found in column {col}")
        except Exception as e:
            logging.error(f"Error visualizing outliers: {e}")
            raise
    def save_results(self, results, file_path):
        logging.info(f"Saving results to {file_path}...")
        try:
            def convert_to_dict(obj):
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient='records')
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, dict):
                    return {k: convert_to_dict(v) for k, v in obj.items()}
                else:
                    return obj

            results_dict = convert_to_dict(results)
            with open(file_path, 'w') as file:
                json.dump(results_dict, file, default=str, indent=4)
            logging.info("Results saved successfully.")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            raise