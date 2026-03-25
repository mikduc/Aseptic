"""Utility functions for file type detection and data handling."""
import os
from pathlib import Path
from typing import Literal
from enum import Enum


class FileType(str, Enum):
    """Supported file types."""
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    PARQUET = "parquet"
    JSONL = "jsonl"


def detect_file_type(file_path: str) -> FileType:
    """
    Detect file type based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        FileType enum value
        
    Raises:
        ValueError: If file type is not supported
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    extension_map = {
        ".csv": FileType.CSV,
        ".json": FileType.JSON,
        ".jsonl": FileType.JSONL,
        ".xlsx": FileType.EXCEL,
        ".xls": FileType.EXCEL,
        ".parquet": FileType.PARQUET,
        ".pq": FileType.PARQUET,
    }
    
    if extension not in extension_map:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(extension_map.keys())}"
        )
    
    return extension_map[extension]


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)
