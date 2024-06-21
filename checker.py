import pandas as pd
import numpy as np

class DataQualityChecker:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def check_missing_values(self):
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        missing_report = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
        return missing_report

    def check_duplicates(self):
        duplicate_rows = self.df[self.df.duplicated()]
        return duplicate_rows

    def check_data_types(self):
        data_types = self.df.dtypes
        return data_types

    def check_outliers(self, threshold=1.5):
        numeric_cols = self.df.select_dtypes(include=[np.number])
        outlier_report = {}
        for col in numeric_cols.columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_report[col] = outliers
        return outlier_report

    def check_value_ranges(self, column_ranges):
        range_report = {}
        for col, (min_val, max_val) in column_ranges.items():
            out_of_range = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)]
            range_report[col] = out_of_range
        return range_report

    def run_all_checks(self, column_ranges=None):
        checks = {}
        checks['Missing Values'] = self.check_missing_values()
        checks['Duplicates'] = self.check_duplicates()
        checks['Data Types'] = self.check_data_types()
        checks['Outliers'] = self.check_outliers()
        if column_ranges:
            checks['Range Validation'] = self.check_value_ranges(column_ranges)
        return checks
