import time
import asyncio
import aiohttp
import pandas as pd
import dask.dataframe as dd
import logging
import matplotlib.pyplot as plt
import json
from openai import OpenAI  
from aiohttp import ClientError
from colored import fg, attr

try:
    import cudf
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataQualityChecker:
    def __init__(self, df, npartitions=4, threshold=1.5, llm_api_key=None):
        self.npartitions = npartitions
        self.threshold = threshold
        self.df = self._initialize_df(df)
        self.is_gpu = HAS_GPU
        self.llm_api_key = llm_api_key
        if llm_api_key:
            self.llm_client = OpenAI(
                base_url='http://localhost:11434/v1',
                api_key=llm_api_key
            )

    def _initialize_df(self, df):
        if HAS_GPU:
            logging.info(f"{fg('green')}Initializing with cuDF for GPU acceleration.{attr('reset')}")
            if isinstance(df, pd.DataFrame):
                return cudf.DataFrame.from_pandas(df)
            elif isinstance(df, cudf.DataFrame):
                return df
            else:
                raise ValueError("Input should be a pandas or cuDF DataFrame")
        else:
            logging.info(f"{fg('green')}Initializing with Dask for parallel processing.{attr('reset')}")
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
        start_time = time.time()
        logging.info(f"{fg('blue')}Checking for missing values...{attr('reset')}")
        try:
            if self.is_gpu:
                missing_values = self.df.isnull().sum()
                missing_percentage = (missing_values / len(self.df)) * 100
                result = cudf.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage}).to_pandas()
            else:
                missing_values = self.df.isnull().sum().compute()
                missing_percentage = (missing_values / len(self.df)) * 100
                result = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
            elapsed_time = time.time() - start_time
            logging.info(f"{fg('blue')}Missing values check completed in {elapsed_time:.2f} seconds.{attr('reset')}")
            return self._print_and_return(result)
        except Exception as e:
            logging.error(f"{fg('red')}Error checking missing values: {e}{attr('reset')}")
            raise

    def check_duplicates(self):
        start_time = time.time()
        logging.info(f"{fg('blue')}Checking for duplicate rows...{attr('reset')}")
        try:
            if self.is_gpu:
                result = self.df[self.df.duplicated()].to_pandas()
            else:
                duplicated_mask = self.df.map_partitions(lambda df: df.duplicated(keep=False))
                result = self.df[duplicated_mask].compute()
            elapsed_time = time.time() - start_time
            logging.info(f"{fg('blue')}Duplicate check completed in {elapsed_time:.2f} seconds.{attr('reset')}")
            return self._print_and_return(result)
        except Exception as e:
            logging.error(f"{fg('red')}Error checking duplicates: {e}{attr('reset')}")
            raise

    def check_data_types(self):
        start_time = time.time()
        logging.info(f"{fg('blue')}Checking data types...{attr('reset')}")
        try:
            data_types = self.df.dtypes
            result = pd.Series(data_types, name='Data Types') if self.is_gpu else data_types
            elapsed_time = time.time() - start_time
            logging.info(f"{fg('blue')}Data types check completed in {elapsed_time:.2f} seconds.{attr('reset')}")
            return self._print_and_return(result)
        except Exception as e:
            logging.error(f"{fg('red')}Error checking data types: {e}{attr('reset')}")
            raise

    def check_outliers(self):
        start_time = time.time()
        logging.info(f"{fg('blue')}Checking for outliers...{attr('reset')}")
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
                logging.info(f"{fg('blue')}Outliers in column {col}:{attr('reset')}")
                logging.info(f"{fg('blue')}{outlier_report[col]}{attr('reset')}")
            elapsed_time = time.time() - start_time
            logging.info(f"{fg('blue')}Outlier check completed in {elapsed_time:.2f} seconds.{attr('reset')}")
            return outlier_report
        except Exception as e:
            logging.error(f"{fg('red')}Error checking outliers: {e}{attr('reset')}")
            raise

    def check_value_ranges(self, column_ranges):
        start_time = time.time()
        logging.info(f"{fg('blue')}Checking for value ranges...{attr('reset')}")
        try:
            range_report = {}
            for col, (min_val, max_val) in column_ranges.items():
                if self.is_gpu:
                    out_of_range = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)]
                    range_report[col] = out_of_range.to_pandas()
                else:
                    out_of_range = self.df[(self.df[col] < min_val) | (self.df[col] > max_val)].compute()
                    range_report[col] = out_of_range
                logging.info(f"{fg('blue')}Values out of range in column {col}:{attr('reset')}")
                logging.info(f"{fg('blue')}{range_report[col]}{attr('reset')}")
            elapsed_time = time.time() - start_time
            logging.info(f"{fg('blue')}Value range check completed in {elapsed_time:.2f} seconds.{attr('reset')}")
            return range_report
        except Exception as e:
            logging.error(f"{fg('red')}Error checking value ranges: {e}{attr('reset')}")
            raise

    async def _call_llm_api(self, session, prompt, model, retries=3):
        url = f"{self.llm_client.base_url}chat/completions".rstrip('/')  # Ensure the URL is correctly formatted
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }

        for attempt in range(retries):
            try:
                async with session.post(url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()
            except (ClientError, aiohttp.ClientResponseError, aiohttp.ServerTimeoutError) as e:
                logging.warning(f"{fg('yellow')}Attempt {attempt + 1} failed: {e}. Retrying...{attr('reset')}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logging.error(f"{fg('red')}Unexpected error: {e}{attr('reset')}")
                raise
        logging.error(f"{fg('red')}All attempts to call the LLM API failed.{attr('reset')}")
        raise ClientError("Failed to call the LLM API after multiple attempts.")

    def generate_summary(self, results):
        summary = "Data Quality Summary Report\n\n"
        
        summary += "Missing Values Summary:\n"
        missing_summary = results['Missing Values'].describe().to_string()
        summary += f"{missing_summary}\n\n"
        
        summary += "Duplicates Summary:\n"
        duplicates_summary = f"Total Duplicates: {results['Duplicates'].shape[0]}"
        summary += f"{duplicates_summary}\n\n"
        
        summary += "Data Types Summary:\n"
        data_types_summary = results['Data Types'].to_string()
        summary += f"{data_types_summary}\n\n"
        
        summary += "Outliers Summary:\n"
        for col, outliers in results['Outliers'].items():
            outliers_summary = f"Outliers in {col}: {outliers.shape[0]}"
            summary += f"{outliers_summary}\n\n"
        
        summary += "Range Validation Summary:\n"
        for col, out_of_range in results['Range Validation'].items():
            range_summary = f"Values out of range in {col}: {out_of_range.shape[0]}"
            summary += f"{range_summary}\n\n"
        
        return summary

    async def generate_llm_report(self, results, model):
        if not self.llm_api_key:
            logging.warning(f"{fg('yellow')}LLM API key is not provided. Skipping LLM report generation.{attr('reset')}")
            return "LLM API key not provided."

        logging.info(f"{fg('blue')}Generating LLM-enhanced report asynchronously...{attr('reset')}")
        try:
            summary = self.generate_summary(results)

            # Use asynchronous HTTP requests to call the LLM API
            async with aiohttp.ClientSession() as session:
                response = await self._call_llm_api(session, summary, model)
                enhanced_report = response['choices'][0]['message']['content']
                return enhanced_report
        except Exception as e:
            logging.error(f"{fg('red')}Error generating LLM-enhanced report asynchronously: {e}{attr('reset')}")
            raise

    def run_all_checks(self, column_ranges=None, llm_model=None):
        start_time = time.time()
        logging.info(f"{fg('blue')}Running all data quality checks...{attr('reset')}")
        try:
            checks = {}
            checks['Missing Values'] = self.check_missing_values()
            checks['Duplicates'] = self.check_duplicates()
            checks['Data Types'] = self.check_data_types()
            checks['Outliers'] = self.check_outliers()
            if column_ranges:
                checks['Range Validation'] = self.check_value_ranges(column_ranges)
            logging.info(f"{fg('blue')}All checks completed.{attr('reset')}")

            total_elapsed_time = time.time() - start_time
            logging.info(f"{fg('green')}Total time for all checks: {total_elapsed_time:.2f} seconds.{attr('reset')}")

            if llm_model:
                loop = asyncio.get_event_loop()
                llm_report = loop.run_until_complete(self.generate_llm_report(checks, model=llm_model))
                print(f"\n{fg('green')}LLM-Enhanced Report:\n{llm_report}{attr('reset')}")
            
            return checks
        except Exception as e:
            logging.error(f"{fg('red')}Error running all checks: {e}{attr('reset')}")
            raise

    def visualize_outliers(self, outlier_report):
        logging.info(f"{fg('blue')}Visualizing outliers...{attr('reset')}")
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
                    logging.info(f"{fg('blue')}No outliers found in column {col}{attr('reset')}")
        except Exception as e:
            logging.error(f"{fg('red')}Error visualizing outliers: {e}{attr('reset')}")
            raise

    def save_results(self, results, file_path):
        logging.info(f"{fg('blue')}Saving results to {file_path}...{attr('reset')}")
        try:
            def convert_to_serializable(obj):
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient='records')
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                else:
                    return obj

            results_serializable = convert_to_serializable(results)
            with open(file_path, 'w') as file:
                json.dump(results_serializable, file, default=str, indent=4)
            logging.info(f"{fg('green')}Results saved successfully.{attr('reset')}")
        except Exception as e:
            logging.error(f"{fg('red')}Error saving results: {e}{attr('reset')}")
            raise

