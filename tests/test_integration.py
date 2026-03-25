"""
Integration tests for CleanSlate modules.
"""
import pytest
import polars as pl
from pathlib import Path
import tempfile
import json

# Import modules
from backend.modules import (
    IntelligentIngestor,
    DataProfiler,
    SafeExecutor,
)
from backend.utils import FileType, detect_file_type


class TestIngestor:
    """Test the Intelligent Ingestor module."""

    def test_file_type_detection(self):
        """Test file type detection."""
        assert detect_file_type("data.csv") == FileType.CSV
        assert detect_file_type("data.json") == FileType.JSON
        assert detect_file_type("data.xlsx") == FileType.EXCEL
        assert detect_file_type("data.parquet") == FileType.PARQUET

    def test_csv_ingestion(self):
        """Test CSV file ingestion."""
        # Create test CSV
        test_df = pl.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "email": ["alice@test.com", "bob@test.com", "charlie@test.com"],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.csv"
            test_df.write_csv(test_file)
            
            ingestor = IntelligentIngestor()
            loaded_df = ingestor.ingest(str(test_file))
            
            assert loaded_df.shape == test_df.shape
            assert loaded_df.columns == test_df.columns

    def test_schema_extraction(self):
        """Test schema extraction."""
        test_df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "score": [0.5, 0.8, 0.9],
        })
        
        ingestor = IntelligentIngestor()
        schema = ingestor.extract_schema(test_df)
        
        assert schema["shape"]["rows"] == 3
        assert schema["shape"]["columns"] == 3
        assert set(schema["columns"]) == {"id", "name", "score"}

    def test_preview_generation(self):
        """Test preview generation."""
        test_df = pl.DataFrame({
            "col": list(range(100))
        })
        
        ingestor = IntelligentIngestor()
        preview = ingestor.get_preview(test_df, n_rows=5)
        
        assert preview.height == 5


class TestProfiler:
    """Test the Data Profiler module."""

    def test_numeric_column_profiling(self):
        """Test profiling of numeric columns."""
        test_df = pl.DataFrame({
            "values": [1, 2, 3, 4, 5, None]
        })
        
        profiler = DataProfiler()
        profile = profiler.profile_column(test_df, "values")
        
        assert profile["column"] == "values"
        assert profile["null_count"] == 1
        assert profile["unique_count"] == 5
        assert "mean" in profile
        assert "median" in profile

    def test_string_column_profiling(self):
        """Test profiling of string columns."""
        test_df = pl.DataFrame({
            "names": ["Alice", "Bob", "Charlie", None, ""]
        })
        
        profiler = DataProfiler()
        profile = profiler.profile_column(test_df, "names")
        
        assert profile["column"] == "names"
        assert profile["null_count"] == 1
        assert profile["empty_string_count"] == 1
        assert "avg_length" in profile
        assert "max_length" in profile

    def test_full_dataframe_profiling(self):
        """Test profiling of full DataFrame."""
        test_df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["A", "B", "C", "D", "E"],
            "score": [0.5, 0.8, 0.9, 0.6, 0.7],
        })
        
        profiler = DataProfiler()
        profile = profiler.profile_dataframe(test_df)
        
        assert profile["total_rows"] == 5
        assert profile["total_columns"] == 3
        assert len(profile["columns"]) == 3


class TestExecutor:
    """Test the Safe Executor module."""

    def test_simple_cleaning_function(self):
        """Test execution of a simple cleaning function."""
        test_df = pl.DataFrame({
            "name": [" Alice ", " Bob ", " Charlie "]
        })
        
        code = """
def clean_data(df):
    '''Strip whitespace from names'''
    return df.with_columns(
        pl.col('name').str.strip_chars()
    )
"""
        
        executor = SafeExecutor()
        cleaned_df = executor.execute_cleaning_function(test_df, code)
        
        assert cleaned_df["name"].to_list() == ["Alice", "Bob", "Charlie"]

    def test_type_conversion_function(self):
        """Test type conversion in cleaning."""
        test_df = pl.DataFrame({
            "age": ["25", "30", "35"]
        })
        
        code = """
def clean_data(df):
    '''Convert age to integer'''
    return df.with_columns(
        pl.col('age').cast(pl.Int32)
    )
"""
        
        executor = SafeExecutor()
        cleaned_df = executor.execute_cleaning_function(test_df, code)
        
        assert cleaned_df["age"].dtype == pl.Int32

    def test_export_dataset(self):
        """Test dataset export."""
        test_df = pl.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["A", "B", "C"],
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.csv"
            
            executor = SafeExecutor()
            result_path = executor.export_dataset(test_df, str(output_path), "csv")
            
            assert Path(result_path).exists()
            loaded_df = pl.read_csv(result_path)
            assert loaded_df.shape == test_df.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
