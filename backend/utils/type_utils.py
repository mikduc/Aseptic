"""Data type mapping and validation utilities."""
import polars as pl
from typing import Dict, Any
from enum import Enum


class DataTypeCategory(str, Enum):
    """Categories of data types for analysis."""
    NUMERIC = "numeric"
    STRING = "string"
    TEMPORAL = "temporal"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


def polars_type_to_string(dtype: pl.DataType) -> str:
    """Convert Polars data type to string representation."""
    return str(dtype)


def categorize_dtype(dtype: pl.DataType) -> DataTypeCategory:
    """
    Categorize a Polars data type.
    
    Args:
        dtype: Polars data type
        
    Returns:
        DataTypeCategory
    """
    dtype_str = str(dtype).lower()
    
    if any(x in dtype_str for x in ["int", "float", "u64", "i64", "decimal"]):
        return DataTypeCategory.NUMERIC
    elif any(x in dtype_str for x in ["str", "string", "utf8"]):
        return DataTypeCategory.STRING
    elif any(x in dtype_str for x in ["date", "time", "datetime", "duration"]):
        return DataTypeCategory.TEMPORAL
    elif "bool" in dtype_str:
        return DataTypeCategory.BOOLEAN
    else:
        return DataTypeCategory.UNKNOWN


def infer_dtype_from_sample(values: list) -> DataTypeCategory:
    """Infer data type from a list of sample values."""
    non_none_values = [v for v in values if v is not None and v != ""]
    
    if not non_none_values:
        return DataTypeCategory.UNKNOWN
    
    first_val = non_none_values[0]
    
    if isinstance(first_val, bool):
        return DataTypeCategory.BOOLEAN
    elif isinstance(first_val, (int, float)):
        return DataTypeCategory.NUMERIC
    elif isinstance(first_val, str):
        # Try to detect if it's a date/time
        return DataTypeCategory.STRING
    else:
        return DataTypeCategory.UNKNOWN
