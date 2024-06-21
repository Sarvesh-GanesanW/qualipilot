import pandas as pd
import dask.dataframe as dd

try:
    import cudf
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

class DataQualityChecker:
    def __init__(self, df):
        if HAS_GPU:
            print("Initializing with cuDF for GPU acceleration.")
            if isinstance(df, pd.DataFrame):
                self.df = cudf.DataFrame.from_pandas(df)
            elif isinstance(df, cudf.DataFrame):
                self.df = df
            else:
                raise ValueError("Input should be a pandas or cuDF DataFrame")
        else:
            print("Initializing with Dask for parallel processing.")
            if isinstance(df, pd.DataFrame):
                self.df = dd.from_pandas(df, npartitions=4)
            elif isinstance(df, dd.DataFrame):
                self.df = df
            else:
                raise ValueError("Input should be a pandas or dask DataFrame")

    def check_missing_values(self):
        print("Checking for missing values...")
        if HAS_GPU:
            missing_values = self.df.isnull().sum()
            missing_percentage = (missing_values / len(self.df)) * 100
            missing_report = cudf.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
            result = missing_report.to_pandas()
        else:
            missing_values = self.df.isnull().sum().compute()
            missing_percentage = (missing_values / len(self.df)) * 100
            missing_report = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
            result = missing_report
        print(result)
        return result

    def check_duplicates(self):
        print("Checking for duplicate rows...")
        if HAS_GPU:
            duplicate_rows = self.df[self.df.duplicated()]
            result = duplicate_rows.to_pandas()
        else:
            duplicated_mask = self.df.map_partitions(lambda df: df.duplicated(keep=False))
            duplicate_rows = self.df[duplicated_mask].compute()
            result = duplicate_rows
        print(result)
        return result

    def check_data_types(self):
        print("Checking data types...")
        if HAS_GPU:
            data_types = self.df.dtypes
            result = pd.Series(data_types, name='Data Types')
        else:
            data_types = self.df.dtypes
            result = data_types
        print(result)
        return result

    def check_outliers(self, threshold=1.5):
        print("Checking for outliers...")
        if HAS_GPU:
            numeric_cols = self.df.select_dtypes(include=['float', 'int'])
            outlier_report = {}
            for col in numeric_cols.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (threshold * IQR)
                upper_bound = Q3 + (threshold * IQR)
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_report[col] = outliers.to_pandas()
        else:
            numeric_cols = self.df.select_dtypes(include=['float', 'int'])
            outlier_report = {}
            for col in numeric_cols.columns:
                Q1 = self.df[col].quantile(0.25).compute()
                Q3 = self.df[col].quantile(0.75).compute()
                IQR = Q3 - Q1
                lower_bound = Q1 - (threshold * IQR)
                upper_bound = Q3 + (threshold * IQR)
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)].compute()
                outlier_report[col] = outliers
        for col, outliers in outlier_report.items():
            print(f"Outliers in column {col}:")
            print(outliers)
        return outlier_report

    def check_value_ranges(self, column_ranges):
        print("Checking for value ranges...")
        if HAS_GPU:
            range_report = {}
            for col, (min_val, max_val) in column_ranges.items():
                out_of_range = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)]
                range_report[col] = out_of_range.to_pandas()
        else:
            range_report = {}
            for col, (min_val, max_val) in column_ranges.items():
                out_of_range = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)].compute()
                range_report[col] = out_of_range
        for col, out_of_range in range_report.items():
            print(f"Values out of range in column {col}:")
            print(out_of_range)
        return range_report

    def run_all_checks(self, column_ranges=None):
        print("Running all data quality checks...")
        checks = {}
        checks['Missing Values'] = self.check_missing_values()
        checks['Duplicates'] = self.check_duplicates()
        checks['Data Types'] = self.check_data_types()
        checks['Outliers'] = self.check_outliers()
        if column_ranges:
            checks['Range Validation'] = self.check_value_ranges(column_ranges)
        print("All checks completed.")
        return checks
