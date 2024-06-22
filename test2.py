import pandas as pd
from datapilot import DataQualityChecker


df = pd.read_csv('Online Sales Data.csv')
checker = DataQualityChecker(df)
checker.set_llm_api_key('ollama')
results = checker.run_all_checks(column_ranges={'Total Revenue': (10, 100)}, llm_model='qwen2:7b')
checker.visualize_outliers(results['Outliers'])
checker.save_results(results, 'results.json')

