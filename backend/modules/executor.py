"""Module E: Executor - Safe code execution and export functionality."""
import polars as pl
from typing import Dict, Any, List, Optional
from pathlib import Path
import tempfile
import multiprocessing as mp
import queue

from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safer_getattr
from RestrictedPython.Eval import default_guarded_getitem
from RestrictedPython import safe_builtins
from config.settings import settings


class ExecutorException(Exception):
    """Custom exception for executor errors."""
    pass


def _run_restricted_cleaner_worker(
    result_queue: mp.Queue,
    df: pl.DataFrame,
    normalized_code: str,
    function_name: Optional[str],
) -> None:
    """Worker process entrypoint for restricted execution."""
    try:
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

        cleaning_func_name = SafeExecutor._resolve_function_name(restricted_locals, function_name)
        if cleaning_func_name is None:
            result_queue.put({"error": "No callable cleaning function found in provided code"})
            return

        cleaning_func = restricted_locals[cleaning_func_name]
        result = cleaning_func(df.clone())

        if not isinstance(result, pl.DataFrame):
            result_queue.put({"error": f"Function must return a Polars DataFrame, got {type(result)}"})
            return

        result_queue.put({"result": result})
    except Exception as exc:
        result_queue.put({"error": str(exc)})


class SafeExecutor:
    """Safely executes cleaning functions in a sandboxed environment."""

    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
        self.max_execution_seconds = settings.MAX_EXECUTION_SECONDS

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
            return self._run_with_timeout(df, normalized_code, function_name)

        except SyntaxError as e:
            raise ExecutorException(f"Syntax error in cleaning code: {str(e)}")
        except Exception as e:
            raise ExecutorException(f"Execution error: {str(e)}")

    def _run_with_timeout(
        self,
        df: pl.DataFrame,
        normalized_code: str,
        function_name: Optional[str],
    ) -> pl.DataFrame:
        """Run restricted code in a child process with hard timeout."""
        ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_run_restricted_cleaner_worker,
            args=(result_queue, df, normalized_code, function_name),
        )

        process.start()
        process.join(timeout=self.max_execution_seconds)

        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            raise ExecutorException(
                f"Execution exceeded time limit ({self.max_execution_seconds}s)"
            )

        try:
            payload = result_queue.get_nowait()
        except queue.Empty as exc:
            raise ExecutorException("Execution failed without result") from exc

        error = payload.get("error") if isinstance(payload, dict) else None
        if error:
            raise ExecutorException(error)

        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, pl.DataFrame):
            raise ExecutorException("Execution did not return a valid DataFrame")

        return result

    def _normalize_user_code(self, python_code: str) -> str:
        """Strip import statements that are blocked in RestrictedPython."""
        cleaned_lines: List[str] = []
        for line in python_code.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    @staticmethod
    def _resolve_function_name(
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
        return None

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
