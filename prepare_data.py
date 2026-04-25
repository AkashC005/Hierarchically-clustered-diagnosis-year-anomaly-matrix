import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
ZIP_CANDIDATES = [
    DATA_DIR / "NHS Hospital Admissions.zip",
    Path("NHS Hospital Admissions.zip"),
]
PARQUET_PATH = DATA_DIR / "hospital_admissions_tidy.parquet"

AGE_ORDER = [
    "0", "1-4", "5-9", "10-14", "15", "16", "17", "18", "19",
    "20-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79",
    "80-84", "85-89", "90+"
]

CHAPTER_MAP = {
    "A": "Infectious & parasitic diseases",
    "B": "Infectious & parasitic diseases",
    "F": "Mental & behavioural disorders",
    "I": "Circulatory diseases",
    "J": "Respiratory diseases",
    "K": "Digestive diseases",
}

def normalise_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip().lower()

def get_zip_path():
    for p in ZIP_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Zip not found.")

def find_sheet_for_level(xls, level_key):
    for name in xls.sheet_names:
        n = name.lower()
        if "primary" not in n:
            continue
        if level_key == "summary" and "summary" in n:
            return name
        if level_key == "3_char" and ("3 char" in n or "3 character" in n):
            return name
        if level_key == "4_char" and ("4 char" in n or "4 character" in n):
            return name
    raise ValueError(level_key)

def find_header_row(raw):
    for i in range(min(60, len(raw))):
        row_text = " | ".join(normalise_text(v) for v in raw.iloc[i].tolist() if normalise_text(v))
        if "finished consultant episodes" in row_text and ("admissions" in row_text or "finished admission episodes" in row_text):
            return i
    raise ValueError("No header row")

def guess_column_index(header_lookup, candidates):
    for c in candidates:
        if c in header_lookup:
            return header_lookup[c]
    for key, idx in header_lookup.items():
        for c in candidates:
            if key.startswith(c):
                return idx
    return None

def parse_code_description_from_row(row):
    code_pattern_3 = re.compile(r"^[A-Z]\d{2}$")
    code_pattern_4 = re.compile(r"^[A-Z]\d{2}[A-Z0-9]$")
    first_four = [row.iloc[i] if i < len(row) else np.nan for i in range(4)]
    if pd.notna(first_four[0]):
        candidate = str(first_four[0]).strip()
        if code_pattern_3.match(candidate) or code_pattern_4.match(candidate):
            desc = str(first_four[1]).strip() if pd.notna(first_four[1]) else candidate
            return candidate, desc
    for v in first_four:
        if pd.notna(v):
            s = str(v).strip()
            m = re.match(r"^([A-Z]\d{2}[A-Z0-9]?)\s+(.*)$", s)
            if m:
                return m.group(1), m.group(2).strip()
    return None, None

def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    return pd.to_numeric(str(x).replace(",", "").strip(), errors="coerce")

def canonical_age_label(raw_label):
    s = normalise_text(raw_label)
    s = s.replace("(fce)", "").replace("(fae)", "").strip()
    s = s.replace("age ", "").strip()
    return s

def derive_chapter(code, level_key, desc):
    if isinstance(code, str) and len(code) > 0 and code[0].upper() in CHAPTER_MAP:
        return CHAPTER_MAP[code[0].upper()]
    if level_key == "summary":
        return desc if isinstance(desc, str) and desc else "Summary"
    return "Other"

def shorten_label(desc, max_len=38):
    if pd.isna(desc):
        return ""
    desc = re.sub(r"\s+", " ", str(desc)).strip()
    return desc if len(desc) <= max_len else desc[: max_len - 1].rstrip(" ,;:-") + "…"

def main():
    zip_path = get_zip_path()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    extract_dir = DATA_DIR / "unzipped"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    root = extract_dir / "NHS Hospital Admissions"
    files = sorted(root.glob("*.xlsx"))
    records = []

    for fp in files:
        m = re.search(r"(20\d{2})-(\d{2})", fp.name.lower())
        if not m:
            continue
        year = int(m.group(1))
        if year < 2012 or year > 2023:
            continue

        xls = pd.ExcelFile(fp)
        for level_key in ["summary", "3_char", "4_char"]:
            try:
                sheet = find_sheet_for_level(xls, level_key)
            except Exception:
                continue

            raw = pd.read_excel(fp, sheet_name=sheet, header=None)
            header_row = find_header_row(raw)
            header = raw.iloc[header_row]

            header_lookup = {
                normalise_text(v): i
                for i, v in header.items()
                if normalise_text(v)
            }

            admissions_col = guess_column_index(header_lookup, ["admissions", "finished admission episodes"])
            male_col = guess_column_index(header_lookup, ["male"])
            female_col = guess_column_index(header_lookup, ["female"])
            emergency_col = guess_column_index(header_lookup, ["emergency"])
            planned_col = guess_column_index(header_lookup, ["planned"])

            age_cols = {}
            for key, idx in header_lookup.items():
                if key.startswith("age "):
                    age_label = canonical_age_label(key)
                    if age_label in AGE_ORDER:
                        age_cols[age_label] = idx

            data = raw.iloc[header_row + 3 :].copy().reset_index(drop=True)
            for _, row in data.iterrows():
                code, desc = parse_code_description_from_row(row)
                if code is None:
                    continue

                rec = {
                    "year": year,
                    "level": level_key,
                    "diagnosis_code": code,
                    "diagnosis_description": str(desc).strip(),
                    "short_label": shorten_label(desc),
                    "chapter": derive_chapter(code, level_key, str(desc).strip()),
                    "Admissions": clean_numeric(row.iloc[admissions_col]) if admissions_col is not None else np.nan,
                    "Male": clean_numeric(row.iloc[male_col]) if male_col is not None else np.nan,
                    "Female": clean_numeric(row.iloc[female_col]) if female_col is not None else np.nan,
                    "Emergency": clean_numeric(row.iloc[emergency_col]) if emergency_col is not None else np.nan,
                    "Planned": clean_numeric(row.iloc[planned_col]) if planned_col is not None else np.nan,
                }
                for age in AGE_ORDER:
                    rec[f"Age_{age}"] = clean_numeric(row.iloc[age_cols[age]]) if age in age_cols else np.nan

                records.append(rec)

    df = pd.DataFrame(records)
    df = df[~(df["Admissions"].isna() & df["Emergency"].isna() & df["Planned"].isna())].copy()
    df["female_share"] = df["Female"] / (df["Female"] + df["Male"])
    df["male_share"] = df["Male"] / (df["Male"] + df["Female"])
    df["emergency_share"] = df["Emergency"] / df["Admissions"]
    df["planned_share"] = df["Planned"] / df["Admissions"]

    age_count_cols = [f"Age_{a}" for a in AGE_ORDER]
    age_total = df[age_count_cols].sum(axis=1).replace(0, np.nan)
    for age in AGE_ORDER:
        df[f"age_share_{age}"] = df[f"Age_{age}"] / age_total

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Saved parquet to {PARQUET_PATH}")

if __name__ == "__main__":
    main()