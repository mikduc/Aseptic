"""Module A: Intelligent Ingestor - File detection and DataFrame loading."""
import polars as pl
from pathlib import Path
from typing import Optional, Tuple
from backend.utils.file_utils import FileType, detect_file_type, get_file_size_mb
from config.settings import settings


class IngestorException(Exception):
    """Custom exception for ingestor errors."""
    pass


class IntelligentIngestor:
    """Handles file type detection and loading into Polars DataFrames."""

    def __init__(self):
        self.max_file_size_mb = settings.MAX_FILE_SIZE_MB

    def ingest(self, file_path: str) -> pl.DataFrame:
        """
        Detect file type and load into a Polars DataFrame.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Polars DataFrame
            
        Raises:
            IngestorException: If file cannot be loaded
        """
        # Validate file exists
        if not Path(file_path).exists():
            raise IngestorException(f"File not found: {file_path}")

        # Check file size
        file_size = get_file_size_mb(file_path)
        if file_size > self.max_file_size_mb:
            raise IngestorException(
                f"File size {file_size:.2f}MB exceeds limit of {self.max_file_size_mb}MB"
            )

        # Detect and load
        file_type = detect_file_type(file_path)
        return self._load_by_type(file_path, file_type)

    def _load_by_type(self, file_path: str, file_type: FileType) -> pl.DataFrame:
        """Load file based on detected type."""
        try:
            if file_type == FileType.CSV:
                return pl.read_csv(file_path)
            elif file_type == FileType.JSON:
                return pl.read_json(file_path)
            elif file_type == FileType.JSONL:
                return pl.read_ndjson(file_path)
            elif file_type == FileType.EXCEL:
                return pl.read_excel(file_path)
            elif file_type == FileType.PARQUET:
                return pl.read_parquet(file_path)
            else:
                raise IngestorException(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise IngestorException(f"Failed to load {file_type.value} file: {str(e)}")

    def extract_schema(self, df: pl.DataFrame) -> dict:
        """
        Extract schema information from a DataFrame.
        
        Returns:
            Schema dict with column names and types
        """
        return {
            "columns": df.columns,
            "dtypes": {col: str(df.schema[col]) for col in df.columns},
            "shape": {"rows": df.height, "columns": df.width},
        }

    def get_preview(self, df: pl.DataFrame, n_rows: int = None) -> pl.DataFrame:
        """
        Get first n rows of the DataFrame as preview.
        
        Args:
            df: Polars DataFrame
            n_rows: Number of rows (defaults to settings.SAMPLE_ROWS)
            
        Returns:
            Preview DataFrame
        """
        if n_rows is None:
            n_rows = settings.SAMPLE_ROWS
        return df.head(n_rows)

    def get_ingest_summary(self, file_path: str) -> dict:
        """
        Get complete ingest summary: schema + preview.
        
        Returns:
            Dict with 'schema', 'preview', and 'file_info'
        """
        df = self.ingest(file_path)
        schema = self.extract_schema(df)
        preview = self.get_preview(df)

        return {
            "file_info": {
                "path": file_path,
                "size_mb": get_file_size_mb(file_path),
                "type": detect_file_type(file_path).value,
            },
            "schema": schema,
            "preview": preview.to_dicts(),
        }
