import numpy as np
import pandas as pd


def compress(data, min_32=False):
    INT8_MIN = np.iinfo(np.int8).min
    INT8_MAX = np.iinfo(np.int8).max
    INT16_MIN = np.iinfo(np.int16).min
    INT16_MAX = np.iinfo(np.int16).max
    INT32_MIN = np.iinfo(np.int32).min
    INT32_MAX = np.iinfo(np.int32).max

    FLOAT16_MIN = np.finfo(np.float16).min
    FLOAT16_MAX = np.finfo(np.float16).max
    FLOAT32_MIN = np.finfo(np.float32).min
    FLOAT32_MAX = np.finfo(np.float32).max
    column_dtypes = {}
    for col in data.columns:
        col_dtype = data[col][:100].dtype

        if pd.api.types.is_numeric_dtype(col_dtype):
            col_series = data[col]
            col_min = col_series.min()
            col_max = col_series.max()

            if pd.api.types.is_float_dtype(col_dtype):
                if (col_min > FLOAT16_MIN) and (col_max < FLOAT16_MAX) and not min_32:
                    column_dtypes[col] = np.float16
                elif (col_min > FLOAT32_MIN) and (col_max < FLOAT32_MAX):
                    column_dtypes[col] = np.float32
                else:
                    column_dtypes[col] = np.float64

            if pd.api.types.is_integer_dtype(col_dtype):
                if (col_min > INT8_MIN / 2) and (col_max < INT8_MAX / 2) and not min_32:
                    column_dtypes[col] = np.int8
                elif (col_min > INT16_MIN) and (col_max < INT16_MAX) and not min_32:
                    column_dtypes[col] = np.int16
                elif (col_min > INT32_MIN) and (col_max < INT32_MAX):
                    column_dtypes[col] = np.int32
                else:
                    column_dtypes[col] = np.int64
    return column_dtypes
