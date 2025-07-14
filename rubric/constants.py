import os
import warnings

RUBRIC_DEFAULT_LLM = os.environ.get("RUBRIC_DEFAULT_LLM", "")
if not RUBRIC_DEFAULT_LLM:
    warnings.warn("RUBRIC_DEFAULT_LLM is not set, some LLM-based functionality may not work.")
