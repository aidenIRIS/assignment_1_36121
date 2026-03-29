"""Data preparation helpers for job-posting datasets."""
import os
import re
import unicodedata
import pandas as pd


def _normalize_text(value):
    """Normalize raw text for stable matching and similarity features."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_acronyms(text):
    """Expand common AI acronyms to improve lexical matching consistency."""
    replacements = {
        r"\bai\b": "artificial intelligence",
        r"\bml\b": "machine learning",
        r"\bnlp\b": "natural language processing",
        r"\bllm\b": "large language model",
        r"\bm l ops\b|\bmlops\b": "machine learning operations",
        r"\bgenai\b": "generative ai",
    }
    out = text
    for pattern, replacement in replacements.items():
        out = re.sub(pattern, replacement, out)
    return out


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}

def load_dataset(path):
    """Load a dataset from a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def preprocess_data(df):
    """
    Preprocess and standardize job-posting data for search/ranking pipelines.

    Steps:
      - remove exact duplicate rows and fully empty rows
      - normalize column names
      - map dataset-specific aliases to canonical columns
      - normalize IT boolean labels
      - create cleaned text columns and a combined clean_text field
      - drop near-duplicates using stable normalized fingerprints
    """
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Canonical schema aliases across datasets.
    alias_map = {
        "title": ["title", "job_title", "position", "position_title"],
        "jobdescription": ["jobdescription", "job_description", "description", "job_desc"],
        "jobrequirement": ["jobrequirement", "jobrequirement", "jobrequirment", "jobrequiment", "requirements"],
        "requiredqual": ["requiredqual", "required_qual", "requiredqualification", "required_qualification", "qualifications"],
        "company": ["company", "company_name", "employer"],
        "it": ["it", "is_it", "ai_related", "is_ai_related"],
    }

    for canonical, aliases in alias_map.items():
        if canonical in df.columns:
            continue
        for col in aliases:
            if col in df.columns:
                df[canonical] = df[col]
                break
        if canonical not in df.columns:
            df[canonical] = ""

    # Backward-compatible alias used across existing scripts.
    df["jobrequiment"] = df["jobrequirement"]

    # Coerce IT filter labels to bool for deterministic filtering.
    df["it"] = df["it"].apply(_coerce_bool)

    # Clean core text columns and expand common acronyms.
    text_cols = ["title", "jobdescription", "jobrequirement", "requiredqual", "company"]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)
        clean_col = f"{col}_clean"
        df[clean_col] = df[col].apply(_normalize_text).apply(_normalize_acronyms)

    # Unified clean text used for similarity/ranking features.
    df["clean_text"] = (
        df["title_clean"]
        + " "
        + df["jobdescription_clean"]
        + " "
        + df["jobrequirement_clean"]
        + " "
        + df["requiredqual_clean"]
    ).str.strip()

    # Remove low-information rows with no usable text.
    df = df[df["clean_text"].str.len() > 0].copy()

    # Near-duplicate pruning via normalized fingerprint.
    df["_dup_key"] = (
        df["title_clean"] + "|" + df["company_clean"] + "|" + df["jobdescription_clean"]
    )
    df = df.drop_duplicates(subset=["_dup_key"]).drop(columns=["_dup_key"])

    return df

# Example usage (to be replaced with actual dataset paths)
# df1 = load_dataset('../data/dataset1.csv')
# df1 = preprocess_data(df1)
# df2 = load_dataset('../data/dataset2.csv')
# df2 = preprocess_data(df2)
