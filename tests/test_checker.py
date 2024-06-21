import pandas as pd
import dask.dataframe as dd
from datapilot import DataQualityChecker

def test_missing_values():
    data = {
        'A': [1, 2, None],
        'B': [4, None, 6]
    }
    df = pd.DataFrame(data)
    checker = DataQualityChecker(df)
    missing_values_report = checker.check_missing_values()
    assert missing_values_report.loc['A', 'Missing Values'] == 1
    assert missing_values_report.loc['B', 'Missing Values'] == 1

def test_duplicates():
    data = {
        'A': [1, 2, 2],
        'B': [4, 5, 5]
    }
    df = pd.DataFrame(data)
    checker = DataQualityChecker(df)
    duplicates = checker.check_duplicates()
    assert len(duplicates) == 1

def test_data_types():
    data = {
        'A': [1, 2, 3],
        'B': ['x', 'y', 'z']
    }
    df = pd.DataFrame(data)
    checker = DataQualityChecker(df)
    data_types = checker.check_data_types()
    assert data_types['A'] == 'int64'
    assert data_types['B'] == 'object'

def test_outliers():
    data = {
        'A': [1, 2, 3, 100],
        'B': [4, 5, 6, 7]
    }
    df = pd.DataFrame(data)
    checker = DataQualityChecker(df)
    outliers = checker.check_outliers()
    assert len(outliers['A']) == 1
    assert len(outliers['B']) == 0

def test_value_ranges():
    data = {
        'A': [1, 2, 3, 10],
        'B': [4, 5, 6, 7]
    }
    df = pd.DataFrame(data)
    checker = DataQualityChecker(df)
    column_ranges = {'A': (0, 5)}
    range_report = checker.check_value_ranges(column_ranges)
    assert len(range_report['A']) == 1

if __name__ == "__main__":
    test_missing_values()
    test_duplicates()
    test_data_types()
    test_outliers()
    test_value_ranges()
    print("All tests passed.")
