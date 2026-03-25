"""Module B: Profiler - Statistical analysis and data quality assessment."""
import polars as pl
from typing import Dict, Any
from datetime import datetime
import sys
from backend.utils.type_utils import categorize_dtype, DataTypeCategory
from config.settings import settings


class ProfilerException(Exception):
    """Custom exception for profiler errors."""
    pass


class DataProfiler:
    """Generates comprehensive data quality profiles for LLM analysis."""

    def __init__(self):
        self.z_score_threshold = 3.0  # Standard deviations for outlier detection
        self.iqr_multiplier = 1.5  # IQR multiplier for outlier detection
        self.ge_supported = sys.version_info < (3, 14)

    def profile_dataframe(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Generate complete data profile for the DataFrame.
        
        Args:
            df: Polars DataFrame
            
        Returns:
            Dict containing comprehensive data profile
        """
        profiles = {}
        health_cards = {}
        for column in df.columns:
            profiles[column] = self.profile_column(df, column)
            health_cards[column] = self.build_data_health_card(df, column, profiles[column])

        card_scores = [card["health_score"] for card in health_cards.values()]
        overall_health_score = round(sum(card_scores) / len(card_scores), 2) if card_scores else 0.0

        return {
            "total_rows": df.height,
            "total_columns": df.width,
            "columns": profiles,
            "health_cards": health_cards,
            "overall_health_score": overall_health_score,
            "metadata": {
                "profiled_at": datetime.now().isoformat(),
                "sample_rows": settings.SAMPLE_ROWS,
            },
        }

    def build_data_health_card(
        self,
        df: pl.DataFrame,
        column: str,
        column_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a GE-backed data health card for a column."""
        ge_module = self._load_ge_module()
        if ge_module is None:
            return self._fallback_health_card(column_profile)

        try:
            ge_df = ge_module.from_pandas(df.select(column).to_pandas())
            checks = []

            null_mostly_threshold = max(0.0, 1.0 - (column_profile.get("null_percentage", 0.0) / 100.0) - 0.01)
            checks.append(
                self._ge_result_to_check(
                    "completeness",
                    ge_df.expect_column_values_to_not_be_null(column, mostly=null_mostly_threshold),
                )
            )

            unique_pct = float(column_profile.get("unique_percentage", 0) or 0)
            if unique_pct >= 95:
                checks.append(
                    self._ge_result_to_check(
                        "uniqueness",
                        ge_df.expect_column_values_to_be_unique(column, mostly=0.95),
                    )
                )

            if column_profile.get("data_type_category") == DataTypeCategory.NUMERIC.value:
                q25 = column_profile.get("q25")
                q75 = column_profile.get("q75")
                if q25 is not None and q75 is not None:
                    iqr = q75 - q25
                    lower = float(q25 - (self.iqr_multiplier * iqr))
                    upper = float(q75 + (self.iqr_multiplier * iqr))
                    checks.append(
                        self._ge_result_to_check(
                            "range_consistency",
                            ge_df.expect_column_values_to_be_between(
                                column,
                                min_value=lower,
                                max_value=upper,
                                mostly=0.95,
                            ),
                        )
                    )

            if column_profile.get("data_type_category") == DataTypeCategory.STRING.value:
                checks.append(
                    {
                        "name": "whitespace_consistency",
                        "passed": not bool(column_profile.get("has_whitespace_issues", False)),
                        "details": {
                            "description": "No leading/trailing whitespace in sampled values",
                        },
                    }
                )

            passed = sum(1 for check in checks if check["passed"])
            total = len(checks)
            score = round((passed / total) * 100, 2) if total else 100.0

            return {
                "status": self._score_to_status(score),
                "health_score": score,
                "checks": checks,
            }
        except Exception:
            return self._fallback_health_card(column_profile)

    def _load_ge_module(self):
        """Safely load Great Expectations only on supported Python runtimes."""
        if not self.ge_supported:
            return None

        try:
            import great_expectations as ge_module
            return ge_module
        except Exception:
            return None

    def _ge_result_to_check(self, name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Great Expectations result payload to card check format."""
        result_payload = result.get("result", {}) if isinstance(result, dict) else {}
        return {
            "name": name,
            "passed": bool(result.get("success", False)) if isinstance(result, dict) else False,
            "details": {
                "unexpected_count": result_payload.get("unexpected_count", 0),
                "unexpected_percent": result_payload.get("unexpected_percent", 0),
            },
        }

    def _fallback_health_card(self, column_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback health card when GE is unavailable."""
        checks = []
        null_pct = float(column_profile.get("null_percentage", 0) or 0)
        checks.append(
            {
                "name": "completeness",
                "passed": null_pct <= 5,
                "details": {"null_percentage": null_pct},
            }
        )

        if column_profile.get("data_type_category") == DataTypeCategory.STRING.value:
            checks.append(
                {
                    "name": "whitespace_consistency",
                    "passed": not bool(column_profile.get("has_whitespace_issues", False)),
                    "details": {},
                }
            )

        passed = sum(1 for check in checks if check["passed"])
        total = len(checks)
        score = round((passed / total) * 100, 2) if total else 100.0

        return {
            "status": self._score_to_status(score),
            "health_score": score,
            "checks": checks,
        }

    def _score_to_status(self, score: float) -> str:
        """Map numeric score to status label."""
        if score >= 85:
            return "healthy"
        if score >= 60:
            return "warning"
        return "critical"

    def profile_column(self, df: pl.DataFrame, column: str) -> Dict[str, Any]:
        """
        Profile a single column.
        
        Args:
            df: Polars DataFrame
            column: Column name
            
        Returns:
            Dict with column profile
        """
        series = df[column]
        dtype = series.dtype
        dtype_category = categorize_dtype(dtype)

        # Basic statistics
        profile = {
            "column": column,
            "data_type": str(dtype),
            "data_type_category": dtype_category.value,
            "null_count": series.is_null().sum(),
            "null_percentage": (series.is_null().sum() / len(series) * 100),
            "unique_count": series.n_unique(),
            "unique_percentage": (series.n_unique() / len(series) * 100),
        }

        # Type-specific statistics
        if dtype_category == DataTypeCategory.NUMERIC:
            profile.update(self._numeric_stats(series))
        elif dtype_category == DataTypeCategory.STRING:
            profile.update(self._string_stats(series))
        elif dtype_category == DataTypeCategory.TEMPORAL:
            profile.update(self._temporal_stats(series))
        elif dtype_category == DataTypeCategory.BOOLEAN:
            profile.update(self._boolean_stats(series))

        # Add sample values
        profile["sample_values"] = series.drop_nulls().head(5).to_list()

        return profile

    def _numeric_stats(self, series: pl.Series) -> Dict[str, Any]:
        """Calculate statistics for numeric columns."""
        non_null = series.drop_nulls()
        
        stats = {
            "mean": float(non_null.mean()) if len(non_null) > 0 else None,
            "median": float(non_null.median()) if len(non_null) > 0 else None,
            "stddev": float(non_null.std()) if len(non_null) > 0 else None,
            "min": float(non_null.min()) if len(non_null) > 0 else None,
            "max": float(non_null.max()) if len(non_null) > 0 else None,
            "q25": float(non_null.quantile(0.25)) if len(non_null) > 0 else None,
            "q75": float(non_null.quantile(0.75)) if len(non_null) > 0 else None,
        }

        # Calculate outliers
        stats["outliers"] = self._detect_numeric_outliers(non_null, stats)
        
        return stats

    def _detect_numeric_outliers(self, series: pl.Series, stats: Dict) -> Dict[str, Any]:
        """Detect outliers using both Z-score and IQR methods."""
        if stats["stddev"] is None or stats["stddev"] == 0:
            return {"z_score_outliers": 0, "iqr_outliers": 0}

        # Z-score method
        mean = stats["mean"]
        stddev = stats["stddev"]
        z_scores = ((series - mean) / stddev).abs()
        z_outliers = (z_scores > self.z_score_threshold).sum()

        # IQR method
        q25 = stats["q25"]
        q75 = stats["q75"]
        iqr = q75 - q25
        lower_bound = q25 - (self.iqr_multiplier * iqr)
        upper_bound = q75 + (self.iqr_multiplier * iqr)
        
        iqr_outliers = ((series < lower_bound) | (series > upper_bound)).sum()

        return {
            "z_score_outliers": int(z_outliers),
            "iqr_outliers": int(iqr_outliers),
            "iqr_bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
        }

    def _string_stats(self, series: pl.Series) -> Dict[str, Any]:
        """Calculate statistics for string columns."""
        non_null = series.drop_nulls()
        
        if len(non_null) == 0:
            return {
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0,
                "empty_string_count": 0,
            }

        lengths = non_null.str.lengths()
        empty_count = (series == "").sum()

        return {
            "avg_length": float(lengths.mean()),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "empty_string_count": int(empty_count),
            "has_special_chars": self._check_special_chars(non_null),
            "has_whitespace_issues": self._check_whitespace_issues(non_null),
        }

    def _check_special_chars(self, series: pl.Series) -> bool:
        """Check if series contains special characters."""
        import re
        special_char_pattern = r"[^a-zA-Z0-9\s_.-]"
        for val in series.head(100):  # Sample check
            if re.search(special_char_pattern, str(val)):
                return True
        return False

    def _check_whitespace_issues(self, series: pl.Series) -> bool:
        """Check if series has leading/trailing whitespace."""
        for val in series.head(100):  # Sample check
            if isinstance(val, str) and val != val.strip():
                return True
        return False

    def _temporal_stats(self, series: pl.Series) -> Dict[str, Any]:
        """Calculate statistics for temporal columns."""
        non_null = series.drop_nulls()
        
        return {
            "earliest": str(non_null.min()) if len(non_null) > 0 else None,
            "latest": str(non_null.max()) if len(non_null) > 0 else None,
            "format_consistency": "unknown",  # TODO: Implement format detection
        }

    def _boolean_stats(self, series: pl.Series) -> Dict[str, Any]:
        """Calculate statistics for boolean columns."""
        non_null = series.drop_nulls()
        true_count = (non_null).sum()
        false_count = len(non_null) - true_count

        return {
            "true_count": int(true_count),
            "false_count": int(false_count),
            "true_percentage": (true_count / len(non_null) * 100) if len(non_null) > 0 else 0,
        }
