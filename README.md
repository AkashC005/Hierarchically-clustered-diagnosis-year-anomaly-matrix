# Interactive Hospital Admissions Anomaly Matrix

This project presents an interactive visualization of NHS hospital admissions data. It is designed to explore how diagnosis-level admissions change over time and whether diagnoses that appear to recover in total admissions also return to their pre-shock age structure.

## What the app shows
- A **diagnosis × year anomaly matrix** relative to the **2015–2019 baseline**
- **Baseline vs recovery age-profile sidecars**
- **Emergency share** across baseline, shock, and recovery
- **Female share** across baseline, shock, and recovery

## Main analytical idea
The app is designed to reveal diagnoses where:
- total admissions move back towards baseline
- but the internal age profile remains different
- suggesting apparent recovery in totals without recovery in case mix

## Features
- 3-character and 4-character diagnosis hierarchy
- Year-range filtering
- Diagnosis chapter filtering
- Diagnosis search
- Gender composition filter
- Admission mode filter
- Focus diagnosis selection
- Responsive desktop / compact layouts
- Additional advanced views:
  - Outlier landscape
  - Age-shift heatmap
  - Cohort fingerprints

## Files
- `app.py` — main Streamlit app
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — theme configuration
- `prepare_data.py` — optional preprocessing script
- `data/NHS Hospital Admissions.zip` — source dataset
- `data/hospital_admissions_tidy.parquet` — optional generated cache

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py