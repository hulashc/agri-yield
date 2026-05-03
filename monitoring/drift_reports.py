from __future__ import annotations

from pathlib import Path
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset, RegressionPreset

REPORT_DIR = Path("data/monitoring/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_reference_and_current(
    reference_path: str,
    current_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_df = pd.read_parquet(reference_path)
    current_df = pd.read_parquet(current_path)
    return reference_df, current_df


def run_data_drift_report(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, report_name: str
):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    html_path = REPORT_DIR / f"{report_name}_data_drift.html"
    json_path = REPORT_DIR / f"{report_name}_data_drift.json"

    report.save_html(str(html_path))
    report.save(str(json_path))
    return html_path, json_path


def run_concept_drift_report(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, report_name: str
):
    # Assumes both dataframes contain target and prediction columns
    report = Report(metrics=[RegressionPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    html_path = REPORT_DIR / f"{report_name}_concept_drift.html"
    json_path = REPORT_DIR / f"{report_name}_concept_drift.json"

    report.save_html(str(html_path))
    report.save(str(json_path))
    return html_path, json_path
