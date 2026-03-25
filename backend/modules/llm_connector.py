"""Module C: LLM Connector - Suggestion Engine for data cleaning recommendations."""
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, TypeAdapter
from backend.modules.llm_prompts import (
    CLEANING_SUGGESTION_SYSTEM_PROMPT,
    create_analysis_prompt,
)
from config.settings import settings

try:
    import openai
except ImportError:
    openai = None

try:
    import litellm
except ImportError:
    litellm = None

try:
    import instructor
except ImportError:
    instructor = None


class CleaningSuggestion(BaseModel):
    """Model for a single cleaning suggestion."""
    column: str
    issue: str
    fix_description: str
    python_code: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class CleaningTaskList(BaseModel):
    """Strict wrapper model used by Instructor for schema-guaranteed responses."""
    tasks: List[CleaningSuggestion]


class SuggestionEngineException(Exception):
    """Custom exception for suggestion engine errors."""
    pass


class SuggestionEngine:
    """Generates data cleaning suggestions using LLM."""

    def __init__(
        self,
        provider: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        self.openai_model = openai_model or settings.OPENAI_MODEL
        self.ollama_base_url = ollama_base_url or settings.OLLAMA_BASE_URL
        self.ollama_model = ollama_model or settings.OLLAMA_MODEL
        self._validate_provider()

    def _validate_provider(self):
        """Validate that required LLM libraries are available."""
        if self.provider == "openai" and not openai:
            raise SuggestionEngineException(
                "OpenAI provider selected but 'openai' package not installed"
            )
        if self.provider == "openai" and not instructor:
            raise SuggestionEngineException(
                "OpenAI provider selected but 'instructor' package not installed"
            )
        if self.provider == "ollama" and not litellm:
            raise SuggestionEngineException(
                "Ollama provider selected but 'litellm' package not installed"
            )

    def generate_suggestions(
        self,
        profile: Dict[str, Any],
        custom_instructions: Optional[str] = None,
    ) -> List[CleaningSuggestion]:
        """
        Generate cleaning suggestions based on data profile.
        
        Args:
            profile: Data profile from DataProfiler
            custom_instructions: Optional additional instructions for the LLM
            
        Returns:
            List of CleaningSuggestion objects
            
        Raises:
            SuggestionEngineException: If LLM call fails
        """
        heuristic_suggestions = self._generate_profile_based_suggestions(profile)

        user_prompt = create_analysis_prompt(profile)
        if custom_instructions:
            user_prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"

        try:
            llm_suggestions = self._call_llm(user_prompt)
            return self._merge_suggestions(heuristic_suggestions, llm_suggestions)
        except Exception as e:
            if heuristic_suggestions:
                return heuristic_suggestions
            raise SuggestionEngineException(
                f"LLM call failed: {str(e)}"
            )

    def _merge_suggestions(
        self,
        heuristic_suggestions: List[CleaningSuggestion],
        llm_suggestions: List[CleaningSuggestion],
    ) -> List[CleaningSuggestion]:
        """Merge suggestions while deduplicating similar column/issue pairs."""
        merged = list(heuristic_suggestions)
        existing = {
            (s.column.lower().strip(), s.issue.lower().strip())
            for s in heuristic_suggestions
        }

        for suggestion in llm_suggestions:
            key = (suggestion.column.lower().strip(), suggestion.issue.lower().strip())
            if key not in existing:
                merged.append(suggestion)
                existing.add(key)

        return merged

    def _generate_profile_based_suggestions(
        self,
        profile: Dict[str, Any],
    ) -> List[CleaningSuggestion]:
        """Generate deterministic suggestions directly from profile signals."""
        suggestions: List[CleaningSuggestion] = []
        columns = profile.get("columns", {})

        for column_name, column_profile in columns.items():
            null_pct = float(column_profile.get("null_percentage", 0) or 0)
            if null_pct >= 5:
                suggestions.append(
                    CleaningSuggestion(
                        column=column_name,
                        issue=f"High missing-value rate ({null_pct:.1f}%)",
                        fix_description="Fill nulls with a safe default for the column type and preserve non-null values.",
                        python_code=(
                            f"def clean_{self._safe_function_name(column_name)}_nulls(df):\n"
                            f"    return df.with_columns(\n"
                            f"        pl.when(pl.col('{column_name}').is_null())\n"
                            f"        .then(pl.col('{column_name}').forward_fill().backward_fill())\n"
                            f"        .otherwise(pl.col('{column_name}'))\n"
                            f"        .alias('{column_name}')\n"
                            f"    )"
                        ),
                        confidence_score=0.86,
                    )
                )

            if column_profile.get("data_type_category") == "string":
                if bool(column_profile.get("has_whitespace_issues", False)):
                    suggestions.append(
                        CleaningSuggestion(
                            column=column_name,
                            issue="Leading or trailing whitespace detected",
                            fix_description="Trim whitespace to normalize text values and reduce accidental duplicates.",
                            python_code=(
                                f"def clean_{self._safe_function_name(column_name)}_whitespace(df):\n"
                                f"    return df.with_columns(\n"
                                f"        pl.col('{column_name}').str.strip_chars().alias('{column_name}')\n"
                                f"    )"
                            ),
                            confidence_score=0.91,
                        )
                    )

                empty_count = int(column_profile.get("empty_string_count", 0) or 0)
                if empty_count > 0:
                    suggestions.append(
                        CleaningSuggestion(
                            column=column_name,
                            issue=f"Empty strings found ({empty_count})",
                            fix_description="Convert empty strings to null values so missing data is handled consistently.",
                            python_code=(
                                f"def clean_{self._safe_function_name(column_name)}_empty_strings(df):\n"
                                f"    return df.with_columns(\n"
                                f"        pl.when(pl.col('{column_name}').str.strip_chars() == '')\n"
                                f"        .then(None)\n"
                                f"        .otherwise(pl.col('{column_name}'))\n"
                                f"        .alias('{column_name}')\n"
                                f"    )"
                            ),
                            confidence_score=0.9,
                        )
                    )

            if column_profile.get("data_type_category") == "numeric":
                outliers = column_profile.get("outliers") or {}
                iqr_outliers = int(outliers.get("iqr_outliers", 0) or 0)
                if iqr_outliers > 0:
                    bounds = outliers.get("iqr_bounds", {})
                    lower = bounds.get("lower")
                    upper = bounds.get("upper")
                    if lower is not None and upper is not None:
                        suggestions.append(
                            CleaningSuggestion(
                                column=column_name,
                                issue=f"Potential numeric outliers detected ({iqr_outliers})",
                                fix_description="Cap extreme values to IQR bounds to reduce distortion from anomalous records.",
                                python_code=(
                                    f"def clean_{self._safe_function_name(column_name)}_outliers(df):\n"
                                    f"    return df.with_columns(\n"
                                    f"        pl.col('{column_name}')\n"
                                    f"        .clip(lower_bound={float(lower)}, upper_bound={float(upper)})\n"
                                    f"        .alias('{column_name}')\n"
                                    f"    )"
                                ),
                                confidence_score=0.82,
                            )
                        )

            unique_pct = float(column_profile.get("unique_percentage", 0) or 0)
            if unique_pct >= 98 and null_pct == 0 and column_profile.get("data_type_category") == "string":
                suggestions.append(
                    CleaningSuggestion(
                        column=column_name,
                        issue="Likely identifier column with near-unique values",
                        fix_description="Standardize casing and trim whitespace for stable joins and deduplication workflows.",
                        python_code=(
                            f"def clean_{self._safe_function_name(column_name)}_identifier_format(df):\n"
                            f"    return df.with_columns(\n"
                            f"        pl.col('{column_name}').str.strip_chars().str.to_uppercase().alias('{column_name}')\n"
                            f"    )"
                        ),
                        confidence_score=0.78,
                    )
                )

        return suggestions

    def _safe_function_name(self, column_name: str) -> str:
        """Return a Python-safe function suffix from a column name."""
        normalized = "".join(ch if ch.isalnum() else "_" for ch in column_name.lower())
        normalized = "_".join(part for part in normalized.split("_") if part)
        return normalized or "column"

    def _call_llm(self, user_prompt: str) -> List[CleaningSuggestion]:
        """
        Call the LLM service.
        
        Args:
            user_prompt: User message for the LLM
            
        Returns:
            LLM response text
        """
        if self.provider == "openai":
            return self._call_openai(user_prompt)
        elif self.provider == "ollama":
            return self._call_ollama(user_prompt)
        else:
            raise SuggestionEngineException(f"Unknown provider: {self.provider}")

    def _call_openai(self, user_prompt: str) -> List[CleaningSuggestion]:
        """Call OpenAI API with Instructor to enforce schema-valid output."""
        if not self.openai_api_key:
            raise SuggestionEngineException("OPENAI_API_KEY is required for OpenAI provider")

        client = openai.OpenAI(api_key=self.openai_api_key)
        instructor_client = instructor.from_openai(client, mode=instructor.Mode.JSON)

        response = instructor_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": CLEANING_SUGGESTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_model=CleaningTaskList,
            temperature=0.7,
            top_p=0.9,
        )

        return response.tasks

    def _call_ollama(self, user_prompt: str) -> List[CleaningSuggestion]:
        """Call Ollama via LiteLLM and validate output against task schema."""
        last_error: Optional[Exception] = None
        for _ in range(2):
            try:
                response = litellm.completion(
                    model=f"ollama/{self.ollama_model}",
                    api_base=self.ollama_base_url,
                    messages=[
                        {"role": "system", "content": CLEANING_SUGGESTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.4,
                    top_p=0.9,
                )
                raw_content = response.choices[0].message.content or "[]"
                return self._parse_response(raw_content)
            except Exception as exc:
                last_error = exc

        raise SuggestionEngineException(f"Ollama response validation failed: {last_error}")

    def _parse_response(self, response: str) -> List[CleaningSuggestion]:
        """
        Parse LLM response into CleaningSuggestion objects.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            List of validated CleaningSuggestion objects
        """
        suggestions_data: Any
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict) and "tasks" in parsed:
                suggestions_data = parsed["tasks"]
            else:
                suggestions_data = parsed
        except json.JSONDecodeError:
            obj_start = response.find("{")
            obj_end = response.rfind("}") + 1
            arr_start = response.find("[")
            arr_end = response.rfind("]") + 1

            if obj_start >= 0 and obj_end > obj_start:
                json_str = response[obj_start:obj_end]
                parsed = json.loads(json_str)
                suggestions_data = parsed.get("tasks", parsed)
            elif arr_start >= 0 and arr_end > arr_start:
                json_str = response[arr_start:arr_end]
                suggestions_data = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON object or array found in response", response, 0)

        task_adapter = TypeAdapter(List[CleaningSuggestion])
        return task_adapter.validate_python(suggestions_data)
