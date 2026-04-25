import re
import zipfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Interactive diagnosis–year hospital admissions anomaly matrix",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# App text
# =========================================================
APP_TITLE = "Recovered totals, reconfigured ages: Hierarchically clustered diagnosis-year anomaly matrix"
APP_SUBTITLE = "3-character primary diagnoses, financial years 2012–2023/24. Highlighted rows show diagnoses whose total admissions move back toward baseline while their age profile remains altered."

# =========================================================
# Paths
# =========================================================
DATA_DIR = Path("data")
ZIP_CANDIDATES = [
    DATA_DIR / "NHS Hospital Admissions.zip",
    Path("NHS Hospital Admissions.zip"),
]
PARQUET_PATH = DATA_DIR / "hospital_admissions_tidy.parquet"

# =========================================================
# Constants
# =========================================================
AGE_ORDER = [
    "0", "1-4", "5-9", "10-14", "15", "16", "17", "18", "19",
    "20-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79",
    "80-84", "85-89", "90+"
]

AGE_TICK_POS = [0, 1, 3, 9, 14, 18, 20, 23]
AGE_TICK_LAB = ["0", "1–4", "10–14", "20–24", "45–49", "65–69", "75–79", "90+"]

BASELINE_YEARS = [2015, 2016, 2017, 2018, 2019]
SHOCK_YEARS = [2020, 2021]
RECOVERY_YEARS = [2022, 2023]

CHAPTER_MAP = {
    "A": "Infectious & parasitic diseases",
    "B": "Infectious & parasitic diseases",
    "F": "Mental & behavioural disorders",
    "I": "Circulatory diseases",
    "J": "Respiratory diseases",
    "K": "Digestive diseases",
}

CHAPTER_COLORS = {
    "Infectious & parasitic diseases": "#6f8f7c",
    "Mental & behavioural disorders": "#9a7b8b",
    "Circulatory diseases": "#7d6fa1",
    "Respiratory diseases": "#7f91ab",
    "Digestive diseases": "#a58663",
    "Other": "#8d8d8d",
}

# =========================================================
# Helpers
# =========================================================
def normalise_text(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x)).strip().lower()


def get_zip_path():
    for p in ZIP_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find 'NHS Hospital Admissions.zip'. Place it in ./data or project root."
    )


def find_sheet_for_level(xls, level_key):
    for name in xls.sheet_names:
        n = name.lower()
        if "primary" not in n:
            continue
        if level_key == "3_char" and ("3 char" in n or "3 character" in n):
            return name
        if level_key == "4_char" and ("4 char" in n or "4 character" in n):
            return name
    raise ValueError(f"No sheet found for level {level_key}")


def find_header_row(raw):
    for i in range(min(60, len(raw))):
        row_text = " | ".join(
            normalise_text(v) for v in raw.iloc[i].tolist() if normalise_text(v)
        )
        if "finished consultant episodes" in row_text and (
            "admissions" in row_text or "finished admission episodes" in row_text
        ):
            return i
    raise ValueError("Could not detect header row")


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
    code_pattern_4 = re.compile(r"^[A-Z]\d{2}\.[A-Z0-9]$")

    first_four = [row.iloc[i] if i < len(row) else np.nan for i in range(4)]

    if pd.notna(first_four[0]):
        candidate = str(first_four[0]).strip()
        if code_pattern_3.match(candidate) or code_pattern_4.match(candidate):
            desc = str(first_four[1]).strip() if pd.notna(first_four[1]) else candidate
            return candidate, desc

    for v in first_four:
        if pd.notna(v):
            s = str(v).strip()
            m = re.match(r"^([A-Z]\d{2}(?:\.[A-Z0-9])?)\s+(.*)$", s)
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


def derive_chapter(code):
    if isinstance(code, str) and len(code) > 0:
        first = code[0].upper()
        if first in CHAPTER_MAP:
            return CHAPTER_MAP[first]
    return "Other"


def shorten_label(desc, max_len=38):
    if pd.isna(desc):
        return ""
    desc = re.sub(r"\s+", " ", str(desc)).strip()
    if len(desc) <= max_len:
        return desc
    return desc[: max_len - 1].rstrip(" ,;:-") + "…"


def wrap_label(label, width=28):
    return textwrap.fill(str(label), width=width)


def signed_log2_ratio(x, baseline):
    if pd.isna(x) or pd.isna(baseline) or baseline <= 0 or x <= 0:
        return np.nan
    return float(np.log2(x / baseline))


def js_distance(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.where(np.isnan(p), 0, p)
    q = np.where(np.isnan(q), 0, q)

    if p.sum() <= 0 or q.sum() <= 0:
        return np.nan

    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))


def zscore(s):
    s = s.astype(float)
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def classify_gender_skew(female_share):
    if pd.isna(female_share):
        return "Unknown"
    if female_share >= 0.55:
        return "Female-skewed"
    if female_share <= 0.45:
        return "Male-skewed"
    return "Balanced"


def classify_mode_skew(em_share, pl_share):
    if pd.isna(em_share) and pd.isna(pl_share):
        return "Unknown"
    if pd.notna(em_share) and pd.notna(pl_share):
        if em_share > pl_share:
            return "Emergency-dominant"
        if pl_share > em_share:
            return "Planned-dominant"
    return "Mixed"


# =========================================================
# Data loading
# =========================================================
@st.cache_data(show_spinner=True)
def build_tidy_dataset_from_zip():
    zip_path = get_zip_path()
    extract_dir = DATA_DIR / "unzipped"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    root = extract_dir / "NHS Hospital Admissions"
    files = sorted(root.glob("*.xlsx"))

    if not files:
        raise FileNotFoundError("No yearly .xlsx files found after extraction")

    records = []

    for fp in files:
        m = re.search(r"(20\d{2})-(\d{2})", fp.name.lower())
        if not m:
            continue

        year = int(m.group(1))
        if year < 2012 or year > 2023:
            continue

        xls = pd.ExcelFile(fp)

        for level_key in ["3_char", "4_char"]:
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

            data = raw.iloc[header_row + 3:].copy().reset_index(drop=True)

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
                    "chapter": derive_chapter(code),
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

    if df.empty:
        return df

    df = df[~(df["Admissions"].isna() & df["Emergency"].isna() & df["Planned"].isna())].copy()

    df["female_share"] = df["Female"] / (df["Female"] + df["Male"])
    df["male_share"] = df["Male"] / (df["Male"] + df["Female"])
    df["emergency_share"] = df["Emergency"] / df["Admissions"]
    df["planned_share"] = df["Planned"] / df["Admissions"]

    age_count_cols = [f"Age_{a}" for a in AGE_ORDER]
    age_total = df[age_count_cols].sum(axis=1).replace(0, np.nan)

    for age in AGE_ORDER:
        df[f"age_share_{age}"] = df[f"Age_{age}"] / age_total

    return df


@st.cache_data(show_spinner=True)
def load_data():
    if PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH)

    df = build_tidy_dataset_from_zip()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(PARQUET_PATH, index=False)
    except Exception:
        pass

    return df


# =========================================================
# Metrics
# =========================================================
def build_anomaly_wide(df_level, age_mode="All ages"):
    if df_level.empty:
        return pd.DataFrame()

    years = sorted(df_level["year"].unique())
    rows = []

    for code, g in df_level.groupby("diagnosis_code"):
        if age_mode == "All ages":
            base = g[g["year"].isin(BASELINE_YEARS)]["Admissions"].mean()
            series = {
                y: signed_log2_ratio(
                    g.loc[g["year"] == y, "Admissions"].iloc[0]
                    if not g.loc[g["year"] == y, "Admissions"].empty else np.nan,
                    base,
                )
                for y in years
            }
        else:
            base = g[g["year"].isin(BASELINE_YEARS)][f"Age_{age_mode}"].mean()
            series = {
                y: signed_log2_ratio(
                    g.loc[g["year"] == y, f"Age_{age_mode}"].iloc[0]
                    if not g.loc[g["year"] == y, f"Age_{age_mode}"].empty else np.nan,
                    base,
                )
                for y in years
            }

        row = {"diagnosis_code": code}
        row.update(series)
        rows.append(row)

    return pd.DataFrame(rows).set_index("diagnosis_code")


def prepare_metrics(df_level):
    if df_level.empty:
        return pd.DataFrame(columns=[
            "diagnosis_code", "diagnosis_description", "short_label", "chapter",
            "baseline_adm", "recovery_adm", "recovery_gap", "age_profile_shift",
            "older_age_drift", "emergency_change", "female_change",
            "baseline_female_share", "baseline_emergency_share", "baseline_planned_share",
            "chapter_divergence", "recoveredness", "selection_score", "story_bonus",
            "gender_skew", "mode_skew"
        ])

    rows = []

    for code, g in df_level.groupby("diagnosis_code"):
        label = g["short_label"].dropna().iloc[0] if not g["short_label"].dropna().empty else code
        desc = g["diagnosis_description"].dropna().iloc[0] if not g["diagnosis_description"].dropna().empty else code
        chapter = g["chapter"].dropna().iloc[0] if not g["chapter"].dropna().empty else "Other"

        baseline = g[g["year"].isin(BASELINE_YEARS)]
        recovery = g[g["year"].isin(RECOVERY_YEARS)]

        baseline_adm = baseline["Admissions"].mean()
        recovery_adm = recovery["Admissions"].mean()

        recovery_gap = np.nan
        if pd.notna(baseline_adm) and baseline_adm > 0 and pd.notna(recovery_adm):
            recovery_gap = (recovery_adm - baseline_adm) / baseline_adm

        baseline_age = np.array([baseline[f"age_share_{a}"].mean() for a in AGE_ORDER], dtype=float)
        recovery_age = np.array([recovery[f"age_share_{a}"].mean() for a in AGE_ORDER], dtype=float)

        age_profile_shift = js_distance(baseline_age, recovery_age)

        older_cols = [f"age_share_{a}" for a in ["65-69", "70-74", "75-79", "80-84", "85-89", "90+"]]
        older_age_drift = recovery[older_cols].mean().sum() - baseline[older_cols].mean().sum()

        em_change = recovery["emergency_share"].mean() - baseline["emergency_share"].mean()
        female_change = recovery["female_share"].mean() - baseline["female_share"].mean()

        rows.append({
            "diagnosis_code": code,
            "diagnosis_description": desc,
            "short_label": label,
            "chapter": chapter,
            "baseline_adm": baseline_adm,
            "recovery_adm": recovery_adm,
            "recovery_gap": recovery_gap,
            "age_profile_shift": age_profile_shift,
            "older_age_drift": older_age_drift,
            "emergency_change": em_change,
            "female_change": female_change,
            "baseline_female_share": baseline["female_share"].mean(),
            "baseline_emergency_share": baseline["emergency_share"].mean(),
            "baseline_planned_share": baseline["planned_share"].mean(),
        })

    m = pd.DataFrame(rows)

    anomaly_wide = build_anomaly_wide(df_level, age_mode="All ages")

    div_rows = []
    for chapter, sub in m.groupby("chapter"):
        codes = sub["diagnosis_code"].tolist()
        present = [c for c in codes if c in anomaly_wide.index]
        if not present:
            continue
        chapter_mat = anomaly_wide.loc[present].fillna(0)
        chapter_mean = chapter_mat.mean(axis=0).values
        for c in present:
            div_rows.append({
                "diagnosis_code": c,
                "chapter_divergence": np.linalg.norm(chapter_mat.loc[c].values - chapter_mean)
            })

    div_df = pd.DataFrame(div_rows)
    if not div_df.empty:
        m = m.merge(div_df, on="diagnosis_code", how="left")
    else:
        m["chapter_divergence"] = np.nan

    m["recoveredness"] = 1 / (1 + m["recovery_gap"].abs())

    for col in [
        "baseline_adm", "age_profile_shift", "older_age_drift",
        "emergency_change", "chapter_divergence", "recoveredness"
    ]:
        m[f"{col}_z"] = zscore(m[col])

    m["story_bonus"] = (
        1.6 * (m["recovery_gap"].abs() <= 0.20).astype(int)
        + 1.8 * (m["age_profile_shift"] >= m["age_profile_shift"].quantile(0.70)).astype(int)
        + 0.7 * (m["emergency_change"].abs() >= m["emergency_change"].abs().quantile(0.70)).astype(int)
        + 0.6 * (m["chapter_divergence"] >= m["chapter_divergence"].quantile(0.70)).astype(int)
    )

    m["selection_score"] = (
        1.25 * m["age_profile_shift_z"]
        + 0.50 * m["recoveredness_z"]
        + 0.35 * m["chapter_divergence_z"].fillna(0)
        + 0.30 * m["emergency_change_z"].abs()
        + 0.20 * m["older_age_drift_z"].abs()
        + 0.20 * m["baseline_adm_z"]
        + m["story_bonus"]
    )

    m["gender_skew"] = m["baseline_female_share"].apply(classify_gender_skew)
    m["mode_skew"] = [
        classify_mode_skew(e, p)
        for e, p in zip(m["baseline_emergency_share"], m["baseline_planned_share"])
    ]

    return m


# =========================================================
# Filtering and selection
# =========================================================
def apply_metric_filters(metrics_df, chapter_filter, diagnosis_search, gender_filter, mode_filter):
    m = metrics_df.copy()
    if m.empty:
        return m

    if chapter_filter:
        m = m[m["chapter"].isin(chapter_filter)]

    if diagnosis_search:
        s = diagnosis_search.lower().strip()
        m = m[
            m["diagnosis_code"].astype(str).str.lower().str.contains(s, na=False)
            | m["diagnosis_description"].astype(str).str.lower().str.contains(s, na=False)
            | m["short_label"].astype(str).str.lower().str.contains(s, na=False)
        ]

    if gender_filter != "All":
        m = m[m["gender_skew"] == gender_filter]

    if mode_filter != "All":
        m = m[m["mode_skew"] == mode_filter]

    return m


def select_visible_diagnoses(metrics_filtered, rows_per_group):
    if metrics_filtered.empty:
        return pd.DataFrame()

    selected_frames = []
    for chapter, sub in metrics_filtered.groupby("chapter"):
        selected_frames.append(
            sub.sort_values("selection_score", ascending=False).head(rows_per_group)
        )

    if not selected_frames:
        return pd.DataFrame()

    selected = pd.concat(selected_frames, ignore_index=False).copy()
    selected["display_group"] = selected["chapter"]
    selected = selected.sort_values(["display_group", "selection_score"], ascending=[True, False])
    return selected


def build_row_meta(selected_meta):
    if selected_meta.empty:
        return []

    row_meta = []
    for group in selected_meta["display_group"].dropna().unique().tolist():
        row_meta.append({
            "row_type": "separator",
            "chapter": group,
            "label": group
        })

        sub = selected_meta[selected_meta["display_group"] == group]
        for _, row in sub.iterrows():
            row_meta.append({
                "row_type": "diagnosis",
                "chapter": row["chapter"],
                "diagnosis_code": row["diagnosis_code"],
                "label": f"{row['diagnosis_code']}  {wrap_label(row['short_label'], 26)}",
            })

    return row_meta


# =========================================================
# Panel data
# =========================================================
def build_panel_data(df_level, row_meta, visible_years, age_mode):
    panel = {
        "row_labels": [r["label"] for r in row_meta],
        "chapters": list(dict.fromkeys([r["chapter"] for r in row_meta if r["row_type"] == "separator"])),
        "band_values": [],
        "main_z": [], "main_custom": [],
        "base_z": [], "base_custom": [],
        "rec_z": [], "rec_custom": [],
        "em_z": [], "em_custom": [],
        "fem_z": [], "fem_custom": [],
        "row_meta": row_meta,
        "visible_years": visible_years,
    }

    chapter_to_num = {c: i for i, c in enumerate(panel["chapters"])}

    for meta in row_meta:
        if meta["row_type"] == "separator":
            panel["band_values"].append([chapter_to_num.get(meta["chapter"], np.nan)])
            panel["main_z"].append([np.nan] * len(visible_years))
            panel["main_custom"].append([[None] * 9 for _ in visible_years])

            panel["base_z"].append([np.nan] * len(AGE_ORDER))
            panel["base_custom"].append([[None] * 5 for _ in AGE_ORDER])

            panel["rec_z"].append([np.nan] * len(AGE_ORDER))
            panel["rec_custom"].append([[None] * 5 for _ in AGE_ORDER])

            panel["em_z"].append([np.nan] * 3)
            panel["em_custom"].append([[None] * 5 for _ in range(3)])

            panel["fem_z"].append([np.nan] * 3)
            panel["fem_custom"].append([[None] * 5 for _ in range(3)])
            continue

        code = meta["diagnosis_code"]
        g = df_level[df_level["diagnosis_code"] == code].copy()
        desc = g["diagnosis_description"].dropna().iloc[0] if not g["diagnosis_description"].dropna().empty else code
        chapter = g["chapter"].dropna().iloc[0] if not g["chapter"].dropna().empty else "Other"

        panel["band_values"].append([chapter_to_num.get(meta["chapter"], np.nan)])

        # Main matrix
        if age_mode == "All ages":
            base_val = g[g["year"].isin(BASELINE_YEARS)]["Admissions"].mean()
            vals, custom = [], []
            for y in visible_years:
                sub = g[g["year"] == y]
                raw = sub["Admissions"].iloc[0] if not sub.empty else np.nan
                anom = signed_log2_ratio(raw, base_val)
                em = sub["emergency_share"].iloc[0] if not sub.empty else np.nan
                pl = sub["planned_share"].iloc[0] if not sub.empty else np.nan
                fem = sub["female_share"].iloc[0] if not sub.empty else np.nan
                vals.append(np.clip(anom, -2, 2) if pd.notna(anom) else np.nan)
                custom.append([code, desc, chapter, y, raw, anom, em, pl, fem])
        else:
            base_val = g[g["year"].isin(BASELINE_YEARS)][f"Age_{age_mode}"].mean()
            vals, custom = [], []
            for y in visible_years:
                sub = g[g["year"] == y]
                raw = sub[f"Age_{age_mode}"].iloc[0] if not sub.empty else np.nan
                anom = signed_log2_ratio(raw, base_val)
                em = sub["emergency_share"].iloc[0] if not sub.empty else np.nan
                pl = sub["planned_share"].iloc[0] if not sub.empty else np.nan
                fem = sub["female_share"].iloc[0] if not sub.empty else np.nan
                vals.append(np.clip(anom, -2, 2) if pd.notna(anom) else np.nan)
                custom.append([code, desc, chapter, y, raw, anom, em, pl, fem])

        panel["main_z"].append(vals)
        panel["main_custom"].append(custom)

        # Baseline / recovery age sidecars
        bvals, bcustom = [], []
        rvals, rcustom = [], []
        for age in AGE_ORDER:
            b = g[g["year"].isin(BASELINE_YEARS)][f"age_share_{age}"].mean()
            r = g[g["year"].isin(RECOVERY_YEARS)][f"age_share_{age}"].mean()
            bvals.append(b)
            rvals.append(r)
            bcustom.append([code, desc, "Baseline", age, b])
            rcustom.append([code, desc, "Recovery", age, r])

        panel["base_z"].append(bvals)
        panel["base_custom"].append(bcustom)
        panel["rec_z"].append(rvals)
        panel["rec_custom"].append(rcustom)

        # Emergency / female B-S-R
        periods = [("Baseline", BASELINE_YEARS), ("Shock", SHOCK_YEARS), ("Recovery", RECOVERY_YEARS)]
        evals, ecustom, fvals, fcustom = [], [], [], []
        for pname, yrs in periods:
            ev = g[g["year"].isin(yrs)]["emergency_share"].mean()
            fv = g[g["year"].isin(yrs)]["female_share"].mean()
            evals.append(ev)
            fvals.append(fv)
            ecustom.append([code, desc, pname, "Emergency share", ev])
            fcustom.append([code, desc, pname, "Female share", fv])

        panel["em_z"].append(evals)
        panel["em_custom"].append(ecustom)
        panel["fem_z"].append(fvals)
        panel["fem_custom"].append(fcustom)

    return panel


# =========================================================
# Figure builders
# =========================================================
def make_colors():
    return {
        "main": [[0.0, "#2d6a96"], [0.5, "#f7f4ef"], [1.0, "#b6443f"]],
        "age": [[0.0, "#f7f4ef"], [1.0, "#3f5f7a"]],
        "em": [[0.0, "#f7efe8"], [1.0, "#cc713d"]],
        "fem": [[0.0, "#5f7d9c"], [0.5, "#f7f4ef"], [1.0, "#bd6287"]],
    }


def add_highlight_shapes(fig, row_meta, highlighted_codes, focus_code, years, xref, yref):
    if not years:
        return

    for i, meta in enumerate(row_meta):
        if meta.get("diagnosis_code") in highlighted_codes:
            fig.add_shape(
                type="rect",
                xref=xref,
                yref=yref,
                x0=min(years) - 0.5,
                x1=max(years) + 0.5,
                y0=i - 0.5,
                y1=i + 0.5,
                line=dict(
                    color="#b0463f",
                    width=2 if meta.get("diagnosis_code") == focus_code else 1
                ),
                fillcolor="rgba(0,0,0,0)",
            )


def build_main_explorer_figure(panel, highlighted_codes, focus_code):
    """
    Single combined figure:
      Col1: chapter colour band
      Col2: admissions anomaly by year
      Col3: age profile baseline
      Col4: age profile recovery
      Col5: emergency B/S/R
      Col6: female B/S/R
    """
    colors = make_colors()
    row_meta = panel["row_meta"]
    years = panel["visible_years"]
    n_rows = len(row_meta)

    fig = make_subplots(
        rows=1,
        cols=6,
        shared_yaxes=True,
        horizontal_spacing=0.018,
        column_widths=[0.04, 0.50, 0.155, 0.155, 0.065, 0.065],
        subplot_titles=[
            "",
            "Admissions anomaly by year",
            "Age profile<br>Baseline",
            "Age profile<br>Recovery",
            "Emergency<br>B / S / R",
            "Female<br>B / S / R",
        ],
    )

    # ---- Col 1: Chapter colour band ----
    zmax_band = max(1, len(panel["chapters"]) - 1)
    fig.add_trace(
        go.Heatmap(
            z=np.array(panel["band_values"], dtype=float),
            x=[""],
            y=list(range(n_rows)),
            colorscale=[
                [0.00, "#6f8f7c"],
                [0.20, "#9a7b8b"],
                [0.40, "#7d6fa1"],
                [0.60, "#7f91ab"],
                [0.80, "#a58663"],
                [1.00, "#8d8d8d"],
            ],
            zmin=0,
            zmax=zmax_band,
            showscale=False,
            hoverinfo="skip",
            xgap=0,
            ygap=1,
        ),
        row=1,
        col=1,
    )

    # ---- Col 2: Main anomaly heatmap ----
    fig.add_trace(
        go.Heatmap(
            z=np.array(panel["main_z"], dtype=float),
            x=years,
            y=list(range(n_rows)),
            zmin=-2,
            zmax=2,
            zmid=0,
            colorscale=colors["main"],
            showscale=False,
            customdata=np.array(panel["main_custom"], dtype=object),
            hovertemplate=(
                "<b>%{customdata[0]} %{customdata[1]}</b><br>"
                "Chapter: %{customdata[2]}<br>"
                "Year: %{customdata[3]}<br>"
                "Value: %{customdata[4]:,.0f}<br>"
                "Anomaly: %{customdata[5]:.2f}<br>"
                "Emergency share: %{customdata[6]:.2f}<br>"
                "Planned share: %{customdata[7]:.2f}<br>"
                "Female share: %{customdata[8]:.2f}<extra></extra>"
            ),
            xgap=1,
            ygap=1,
        ),
        row=1,
        col=2,
    )

    # ---- Col 3: Baseline age profile ----
    fig.add_trace(
        go.Heatmap(
            z=np.array(panel["base_z"], dtype=float),
            x=list(range(len(AGE_ORDER))),
            y=list(range(n_rows)),
            zmin=0,
            zmax=0.45,
            colorscale=colors["age"],
            showscale=False,
            customdata=np.array(panel["base_custom"], dtype=object),
            hovertemplate=(
                "<b>%{customdata[0]} %{customdata[1]}</b><br>"
                "Period: %{customdata[2]}<br>"
                "Age band: %{customdata[3]}<br>"
                "Age share: %{customdata[4]:.3f}<extra></extra>"
            ),
            xgap=1,
            ygap=1,
        ),
        row=1,
        col=3,
    )

    # ---- Col 4: Recovery age profile ----
    fig.add_trace(
        go.Heatmap(
            z=np.array(panel["rec_z"], dtype=float),
            x=list(range(len(AGE_ORDER))),
            y=list(range(n_rows)),
            zmin=0,
            zmax=0.45,
            colorscale=colors["age"],
            showscale=False,
            customdata=np.array(panel["rec_custom"], dtype=object),
            hovertemplate=(
                "<b>%{customdata[0]} %{customdata[1]}</b><br>"
                "Period: %{customdata[2]}<br>"
                "Age band: %{customdata[3]}<br>"
                "Age share: %{customdata[4]:.3f}<extra></extra>"
            ),
            xgap=1,
            ygap=1,
        ),
        row=1,
        col=4,
    )

    # ---- Col 5: Emergency share B/S/R ----
    fig.add_trace(
        go.Heatmap(
            z=np.array(panel["em_z"], dtype=float),
            x=["B", "S", "R"],
            y=list(range(n_rows)),
            zmin=0,
            zmax=1,
            colorscale=colors["em"],
            showscale=False,
            customdata=np.array(panel["em_custom"], dtype=object),
            hovertemplate=(
                "<b>%{customdata[0]} %{customdata[1]}</b><br>"
                "Period: %{customdata[2]}<br>"
                "%{customdata[3]}: %{customdata[4]:.3f}<extra></extra>"
            ),
            xgap=1,
            ygap=1,
        ),
        row=1,
        col=5,
    )

    # ---- Col 6: Female share B/S/R ----
    fig.add_trace(
        go.Heatmap(
            z=np.array(panel["fem_z"], dtype=float),
            x=["B", "S", "R"],
            y=list(range(n_rows)),
            zmin=0.35,
            zmax=0.65,
            zmid=0.50,
            colorscale=colors["fem"],
            showscale=False,
            customdata=np.array(panel["fem_custom"], dtype=object),
            hovertemplate=(
                "<b>%{customdata[0]} %{customdata[1]}</b><br>"
                "Period: %{customdata[2]}<br>"
                "%{customdata[3]}: %{customdata[4]:.3f}<extra></extra>"
            ),
            xgap=1,
            ygap=1,
        ),
        row=1,
        col=6,
    )

    # ---- Y-axes ----
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n_rows)),
        ticktext=panel["row_labels"],
        autorange="reversed",
        row=1,
        col=1,
        tickfont=dict(size=10),
        showgrid=False,
    )

    for c in [2, 3, 4, 5, 6]:
        fig.update_yaxes(
            showticklabels=False,
            row=1,
            col=c,
            autorange="reversed",
            showgrid=False,
        )

    # ---- X-axes ----
    fig.update_xaxes(tickfont=dict(size=10), showgrid=False, row=1, col=2)

    for c in [3, 4]:
        fig.update_xaxes(
            tickmode="array",
            tickvals=AGE_TICK_POS,
            ticktext=AGE_TICK_LAB,
            tickfont=dict(size=9),
            showgrid=False,
            row=1,
            col=c,
        )

    for c in [5, 6]:
        fig.update_xaxes(tickfont=dict(size=9), showgrid=False, row=1, col=c)

    # ---- Highlight outlines ----
    add_highlight_shapes(fig, row_meta, highlighted_codes, focus_code, years, xref="x2", yref="y")

    # ---- Subplot title styling ----
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color="#333333")
        ann.y = 1.03

    # =====================================================
    # FIXED COLORBARS
    # =====================================================
    # Do not use visible=False for these. Plotly hides the colorbar
    # when the trace is invisible. Instead, use opacity=0.
    def add_bottom_colorbar(
        colorscale,
        zmin,
        zmax,
        x,
        y,
        length,
        thickness,
        title,
        tickvals,
        ticktext,
        zmid=None,
        tickfont_size=8,
        titlefont_size=8,
    ):
        trace_kwargs = dict(
            z=[[zmin, zmax]],
            x=[0, 1],
            y=[0],
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            showscale=True,
            opacity=0,
            hoverinfo="skip",
            colorbar=dict(
                orientation="h",
                x=x,
                xanchor="center",
                y=y,
                yanchor="top",
                len=length,
                thickness=thickness,
                tickvals=tickvals,
                ticktext=ticktext,
                tickfont=dict(size=tickfont_size),
                title=dict(
                    text=title,
                    side="bottom",
                    font=dict(size=titlefont_size),
                ),
            ),
        )

        if zmid is not None:
            trace_kwargs["zmid"] = zmid

        fig.add_trace(
            go.Heatmap(**trace_kwargs),
            row=1,
            col=2,
        )

    add_bottom_colorbar(
        colorscale=colors["main"],
        zmin=-2,
        zmax=2,
        zmid=0,
        x=0.28,
        y=-0.15,
        length=0.52,
        thickness=13,
        title="Admissions anomaly vs 2015–2019 baseline",
        tickvals=[-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2],
        ticktext=["-2.0", "-1.5", "-1.0", "-0.5", "0.0", "0.5", "1.0", "1.5", "2.0"],
        tickfont_size=9,
        titlefont_size=9,
    )

    add_bottom_colorbar(
        colorscale=colors["age"],
        zmin=0,
        zmax=0.45,
        x=0.665,
        y=-0.15,
        length=0.22,
        thickness=13,
        title="Age share within diagnosis",
        tickvals=[0, 0.10, 0.20, 0.30, 0.40],
        ticktext=["0", "0.10", "0.20", "0.30", "0.40"],
        tickfont_size=8,
        titlefont_size=8,
    )

    add_bottom_colorbar(
        colorscale=colors["em"],
        zmin=0,
        zmax=1,
        x=0.845,
        y=-0.15,
        length=0.075,
        thickness=11,
        title="Emergency share",
        tickvals=[0, 0.5, 1.0],
        ticktext=["0", "0.5", "1.0"],
        tickfont_size=8,
        titlefont_size=8,
    )

    add_bottom_colorbar(
        colorscale=colors["fem"],
        zmin=0.35,
        zmax=0.65,
        zmid=0.50,
        x=0.955,
        y=-0.15,
        length=0.075,
        thickness=11,
        title="Female share",
        tickvals=[0.4, 0.5, 0.6],
        ticktext=["0.4", "0.5", "0.6"],
        tickfont_size=8,
        titlefont_size=8,
    )

    fig.update_layout(
        height=max(780, 34 * n_rows + 260),
        margin=dict(l=10, r=10, t=68, b=175),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=11),
    )

    return fig


def build_outlier_figure(metrics_filtered, focus_code):
    colors = CHAPTER_COLORS.copy()
    fig = go.Figure()

    for chapter, sub in metrics_filtered.groupby("chapter"):
        fig.add_trace(
            go.Scatter(
                x=sub["recovery_gap"],
                y=sub["age_profile_shift"],
                mode="markers+text" if len(sub) <= 12 else "markers",
                text=sub["diagnosis_code"],
                textposition="top center",
                marker=dict(
                    size=np.clip(np.sqrt(sub["baseline_adm"].fillna(0).clip(lower=1)) / 5, 8, 28),
                    color=colors.get(chapter, "#8d8d8d"),
                    line=dict(
                        width=np.where(sub["diagnosis_code"] == focus_code, 2.5, 0.5),
                        color="#333333",
                    ),
                ),
                name=chapter,
                customdata=np.stack(
                    [
                        sub["diagnosis_code"],
                        sub["short_label"],
                        sub["baseline_adm"],
                        sub["emergency_change"],
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]} %{customdata[1]}</b><br>"
                    "Recovery gap: %{x:.2%}<br>"
                    "Age-profile shift: %{y:.3f}<br>"
                    "Baseline admissions: %{customdata[2]:,.0f}<br>"
                    "Emergency Δ: %{customdata[3]:+.3f}<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#999999")

    if not metrics_filtered["age_profile_shift"].dropna().empty:
        fig.add_hline(
            y=metrics_filtered["age_profile_shift"].median(),
            line_width=1,
            line_dash="dot",
            line_color="#bbbbbb",
        )

    fig.update_layout(
        title="Outlier landscape: recovery gap versus age-profile shift",
        xaxis_title="Recovery gap relative to 2015–2019 baseline",
        yaxis_title="Baseline–recovery age-profile shift",
        height=560,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text="Chapter",
        margin=dict(l=20, r=20, t=70, b=50),
    )

    return fig


def build_parallel_coordinates(metrics_filtered):
    if metrics_filtered.empty:
        return go.Figure()

    sub = metrics_filtered.sort_values("selection_score", ascending=False).head(25).copy()
    codes = sub["diagnosis_code"].astype(str).tolist()
    tickvals = list(range(len(codes)))
    color_vals = sub["selection_score"].fillna(0)

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=color_vals,
                colorscale=[[0.0, "#cfd8e3"], [1.0, "#b0463f"]],
                showscale=True,
                colorbar=dict(title="Score"),
            ),
            dimensions=[
                dict(label="Diagnosis", values=tickvals, tickvals=tickvals, ticktext=codes),
                dict(label="Recovery gap", values=sub["recovery_gap"].fillna(0)),
                dict(label="Age shift", values=sub["age_profile_shift"].fillna(0)),
                dict(label="Emergency Δ", values=sub["emergency_change"].fillna(0)),
                dict(label="Female Δ", values=sub["female_change"].fillna(0)),
                dict(label="Older-age drift", values=sub["older_age_drift"].fillna(0)),
            ],
        )
    )

    fig.update_layout(
        title="Cohort fingerprints across multiple metrics",
        height=520,
        margin=dict(l=30, r=30, t=70, b=20),
        paper_bgcolor="white",
    )

    return fig


# =========================================================
# UI
# =========================================================
df = load_data()

if df.empty:
    st.error("No data could be loaded from the NHS workbook zip.")
    st.stop()

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.1rem; padding-bottom: 1rem;}
    [data-testid="stSidebar"] {min-width: 280px; max-width: 320px;}
    .small-note {font-size: 0.92rem; color: #555;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)
st.caption(
    "B = Baseline (2015–2019); S = Shock (2020–2021); R = Recovery (2022–2023/24). "
    "Rows are selected and clustered within chapter to emphasise diagnoses whose "
    "headline recovery masks internal age-profile change."
)

with st.sidebar:
    st.header("Controls")

    level_label = st.radio("Hierarchy level", ["3-character", "4-character"], index=0)
    level_key = {"3-character": "3_char", "4-character": "4_char"}[level_label]

    df_level = df[df["level"] == level_key].copy()
    if df_level.empty:
        st.warning("No data available for this hierarchy level.")
        st.stop()

    year_min = int(df_level["year"].min())
    year_max = int(df_level["year"].max())
    year_range = st.slider(
        "Visible year range",
        min_value=year_min,
        max_value=year_max,
        value=(2012, 2023),
        step=1,
    )

    matrix_mode = st.selectbox(
        "Main matrix mode",
        ["All ages: admissions anomaly", "Age-specific anomaly"],
        index=0,
    )

    age_band = None
    if matrix_mode == "Age-specific anomaly":
        age_band = st.selectbox("Age band", AGE_ORDER, index=14)

    metrics_all = prepare_metrics(df_level)
    if metrics_all.empty:
        st.warning("No diagnosis metrics could be computed for this hierarchy level.")
        st.stop()

    available_chapters = sorted(metrics_all["chapter"].dropna().unique().tolist())
    chapter_filter = st.multiselect(
        "Diagnosis chapter",
        available_chapters,
        default=available_chapters,
    )

    diagnosis_search = st.text_input("Diagnosis search", "")

    gender_filter = st.selectbox(
        "Gender composition filter",
        ["All", "Female-skewed", "Male-skewed", "Balanced"],
        index=0,
    )

    mode_filter = st.selectbox(
        "Admission mode filter",
        ["All", "Emergency-dominant", "Planned-dominant"],
        index=0,
    )

    rows_per_group = st.slider("Rows per chapter/group", 2, 8, 4)

metrics_filtered = apply_metric_filters(
    metrics_all,
    chapter_filter=chapter_filter,
    diagnosis_search=diagnosis_search,
    gender_filter=gender_filter,
    mode_filter=mode_filter,
)

if metrics_filtered.empty:
    st.warning("No diagnoses match the current filters.")
    st.stop()

selected_meta = select_visible_diagnoses(metrics_filtered, rows_per_group)

if selected_meta.empty:
    st.warning("No diagnoses remain after selection.")
    st.stop()

focus_options_df = selected_meta.sort_values("selection_score", ascending=False).copy()
focus_options_df["focus_label"] = (
    focus_options_df["diagnosis_code"].astype(str)
    + " — "
    + focus_options_df["short_label"].astype(str)
)

focus_options = focus_options_df["focus_label"].drop_duplicates().tolist()

if "focus_choice" not in st.session_state or st.session_state["focus_choice"] not in focus_options:
    st.session_state["focus_choice"] = focus_options[0]

with st.sidebar:
    focus_choice = st.selectbox("Focus diagnosis", focus_options, key="focus_choice")

focus_code = focus_choice.split(" — ")[0]

highlighted = selected_meta.sort_values("selection_score", ascending=False).head(3).copy()
highlighted_codes = highlighted["diagnosis_code"].tolist()

row_meta = build_row_meta(selected_meta)
visible_years = list(range(year_range[0], year_range[1] + 1))
age_mode = "All ages" if matrix_mode == "All ages: admissions anomaly" else age_band

panel = build_panel_data(
    df_level[df_level["diagnosis_code"].isin(selected_meta["diagnosis_code"])].copy(),
    row_meta,
    visible_years,
    age_mode,
)

main_tab, outlier_tab, cohort_tab = st.tabs(
    ["Main explorer", "Outlier landscape", "Cohort fingerprints"]
)

with main_tab:
    st.plotly_chart(
        build_main_explorer_figure(panel, highlighted_codes, focus_code),
        use_container_width=True,
        config={"responsive": True, "displaylogo": False},
    )

with outlier_tab:
    st.plotly_chart(
        build_outlier_figure(metrics_filtered, focus_code),
        use_container_width=True,
        config={"responsive": True, "displaylogo": False},
    )

with cohort_tab:
    st.plotly_chart(
        build_parallel_coordinates(metrics_filtered),
        use_container_width=True,
        config={"responsive": True, "displaylogo": False},
    )

focus_row = selected_meta[selected_meta["diagnosis_code"] == focus_code]

if focus_row.empty:
    focus_row = highlighted.head(1)

focus_row = focus_row.iloc[0]

col1, col2 = st.columns([1.0, 1.2], gap="large")

with col1:
    st.markdown("### Focus diagnosis")
    st.markdown(
        f"""
        **{focus_row['diagnosis_code']} {focus_row['short_label']}**  
        Chapter: {focus_row['chapter']}  
        Recovery gap: {focus_row['recovery_gap']:.2%}  
        Age-profile shift: {focus_row['age_profile_shift']:.3f}  
        Emergency-share change: {focus_row['emergency_change']:+.3f}
        """
    )

with col2:
    st.markdown("### Top diagnoses under current filters")
    show_cols = [
        "diagnosis_code",
        "short_label",
        "chapter",
        "selection_score",
        "recovery_gap",
        "age_profile_shift",
        "emergency_change",
    ]

    st.dataframe(
        highlighted[show_cols].rename(
            columns={
                "diagnosis_code": "Code",
                "short_label": "Diagnosis",
                "chapter": "Chapter",
                "selection_score": "Score",
                "recovery_gap": "Recovery gap",
                "age_profile_shift": "Age-profile shift",
                "emergency_change": "Emergency Δ",
            }
        ).round(3),
        use_container_width=True,
        hide_index=True,
    )