"""Module E: Executor - Safe code execution and export functionality."""
import polars as pl
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile

from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safer_getattr
from RestrictedPython.Eval import default_guarded_getitem
from RestrictedPython import safe_builtins


class ExecutorException(Exception):
    """Custom exception for executor errors."""
    pass


class SafeExecutor:
    """Safely executes cleaning functions in a sandboxed environment."""

    def __init__(self):
        self.temp_dir = tempfile.gettempdir()

    def execute_cleaning_function(
        self,
        df: pl.DataFrame,
        python_code: str,
        function_name: Optional[str] = "clean_data",
    ) -> pl.DataFrame:
        """
        Execute a cleaning function safely.
        
        Args:
            df: Input Polars DataFrame
            python_code: Python function code as string
            function_name: Name of the function to call (defaults to 'clean_data')
            
        Returns:
            Cleaned Polars DataFrame
            
        Raises:
            ExecutorException: If execution fails
        """
        try:
            normalized_code = self._normalize_user_code(python_code)

            restricted_globals = {
                "__builtins__": {
                    **safe_builtins,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "list": list,
                    "dict": dict,
                    "print": print,
                    "range": range,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                },
                "pl": pl,
                "_getattr_": safer_getattr,
                "_getitem_": default_guarded_getitem,
                "_getiter_": iter,
            }
            restricted_locals: Dict[str, Any] = {}

            byte_code = compile_restricted(normalized_code, "<cleaning_code>", "exec")
            exec(byte_code, restricted_globals, restricted_locals)

            cleaning_func_name = self._resolve_function_name(restricted_locals, function_name)
            if cleaning_func_name is None:
                raise ExecutorException("No callable cleaning function found in provided code")

            cleaning_func = restricted_locals[cleaning_func_name]

            result = cleaning_func(df.clone())

            # Validate result is a DataFrame
            if not isinstance(result, pl.DataFrame):
                raise ExecutorException(
                    f"Function must return a Polars DataFrame, got {type(result)}"
                )

            return result

        except SyntaxError as e:
            raise ExecutorException(f"Syntax error in cleaning code: {str(e)}")
        except Exception as e:
            raise ExecutorException(f"Execution error: {str(e)}")

    def _normalize_user_code(self, python_code: str) -> str:
        """Strip import statements that are blocked in RestrictedPython."""
        cleaned_lines: List[str] = []
        for line in python_code.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def _resolve_function_name(
        self,
        namespace: Dict[str, Any],
        preferred_name: Optional[str],
    ) -> Optional[str]:
        """Resolve which cleaning function to run from restricted locals."""
        if preferred_name and preferred_name in namespace and callable(namespace[preferred_name]):
            return preferred_name

        if "clean_data" in namespace and callable(namespace["clean_data"]):
            return "clean_data"

        clean_functions = [
            name for name, value in namespace.items()
            if callable(value) and name.startswith("clean_")
        ]
        if clean_functions:
            return clean_functions[0]

        any_callable = [name for name, value in namespace.items() if callable(value)]
        return any_callable[0] if any_callable else None

    def execute_multiple_cleaners(
        self,
        df: pl.DataFrame,
        cleaners: List[Dict[str, str]],
    ) -> pl.DataFrame:
        """
        Execute multiple cleaning functions sequentially.
        
        Args:
            df: Input DataFrame
            cleaners: List of dicts with 'python_code' and optional 'function_name'
            
        Returns:
            Cleaned DataFrame
        """
        result_df = df.clone()
        
        for i, cleaner in enumerate(cleaners):
            code = cleaner.get("python_code")
            func_name = cleaner.get("function_name", f"clean_step_{i}")
            
            if not code:
                raise ExecutorException(f"Cleaner {i} missing 'python_code'")
            
            try:
                result_df = self.execute_cleaning_function(
                    result_df, code, func_name
                )
            except ExecutorException as e:
                raise ExecutorException(f"Failed at cleaner {i}: {str(e)}")
        
        return result_df

    def export_dataset(
        self,
        df: pl.DataFrame,
        output_path: str,
        format: str = "csv",
    ) -> str:
        """
        Export cleaned dataset.
        
        Args:
            df: Cleaned DataFrame
            output_path: Path to save the file
            format: Export format (csv, parquet, json)
            
        Returns:
            Path to exported file
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "csv":
                df.write_csv(str(output_path))
            elif format.lower() == "parquet":
                df.write_parquet(str(output_path))
            elif format.lower() == "json":
                df.write_json(str(output_path))
            else:
                raise ExecutorException(f"Unsupported export format: {format}")
            
            return str(output_path)
        except Exception as e:
            raise ExecutorException(f"Export failed: {str(e)}")

    def generate_cleaning_script(
        self,
        cleaners: List[Dict[str, Any]],
        output_path: str,
    ) -> str:
        """
        Generate a standalone Python cleaning script.
        
        Args:
            cleaners: List of cleaning suggestions
            output_path: Path to save the script
            
        Returns:
            Path to generated script
        """
        script_content = '''"""
Generated Data Cleaning Script
Auto-generated by CleanSlate
"""
import polars as pl
from pathlib import Path


def main(input_file: str, output_file: str) -> None:
    """Main cleaning pipeline."""
    # Load the dataset
    df = pl.read_csv(input_file)
    print(f"Loaded {df.height} rows, {df.width} columns")
    
    # Apply cleaning steps
'''

        for i, cleaner in enumerate(cleaners):
            script_content += f"\n    # Step {i + 1}: {cleaner.get('issue', 'Cleaning step')}\n"
            
            # Extract function from the code
            code_lines = cleaner.get("python_code", "").split("\n")
            indented_code = "\n".join("    " + line for line in code_lines)
            script_content += indented_code + "\n"
            
            func_name = cleaner.get("function_name", f"clean_step_{i}")
            script_content += f"    df = {func_name}(df)\n"
            script_content += f'    print("Step {i + 1} completed")\n'

        script_content += '''
    # Export the cleaned dataset
    df.write_csv(output_file)
    print(f"Saved {df.height} rows to {output_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python cleaning_script.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
'''

        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(script_content)
            return str(output_path)
        except Exception as e:
            raise ExecutorException(f"Failed to generate script: {str(e)}")
