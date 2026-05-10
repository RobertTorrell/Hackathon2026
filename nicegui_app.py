#!/usr/bin/env python3
"""NiceGUI dashboard for Smart Demand Signals.

The app uploads a hackathon workbook, runs the existing analytical pipeline in
project.py, and presents forecasts, validation, alerts, and client drill-downs.
"""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nicegui import app, run, ui
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

import project as sd


APP_PORT = int(os.getenv("APP_PORT", "1982"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
STATE_LABELS = ["Loyal", "Promising", "Risky", "Promiscuous"]
REQUIRED_SHEETS = {"Ventas", "Productos", "Clientes", "Potencial", "Campañas"}
REQUIRED_COLUMNS = {
    "Ventas": {"Num.Fact", "Fecha", "Id. Cliente", "Id. Producto", "Unidades", "Valores_H"},
    "Productos": {"Id.Prod", "Bloque analítico", "Categoria_H", "Familia_H"},
    "Clientes": {"Id. Cliente", "Provincia"},
    "Potencial": {"Id.Cliente", "Familia", "Categoria Productos", "Potencial_H"},
    "Campañas": {"Campaña", "Fecha inicio", "Fecha fin"},
}

ML_FEATURES = [
    "spend_4w",
    "prev_spend_4w",
    "spend_12w",
    "spend_active_window",
    "spend_capture_window",
    "annual_potential",
    "potential_4w",
    "potential_capture_window",
    "capture_ratio",
    "capture_ratio_window",
    "observed_trend_pct",
    "classification_observed_slope",
    "trend_4w_vs_prev_4w",
    "active_weeks_12w",
    "active_weeks_4w",
    "active_weeks_52w",
    "weekly_mean_12w",
    "weekly_std_12w",
    "volatility_12w",
    "weeks_since_last_purchase",
    "regularity_score",
    "abnormal_silence",
    "capture_strength",
    "client_weighted_historical_product_slope",
    "client_weighted_forecast_product_slope",
    "predicted_spend_4w",
    "predicted_capture_ratio",
    "predicted_capture_ratio_window",
    "predicted_capture_strength",
    "predicted_spend_change_pct",
    "forecast_trend_pct",
    "classification_forecast_slope",
    "client_product_forecast_factor",
    "category_forecast_factor",
]


@dataclass
class MlArtifacts:
    model: LogisticRegression
    scaler: StandardScaler
    encoder: LabelEncoder
    coefficients: pd.DataFrame


@dataclass
class AnalysisResult:
    run_dir: Path
    summary: dict[str, Any]
    weekly_sales: pd.DataFrame
    type_forecasts: pd.DataFrame
    product_forecasts: pd.DataFrame
    classifications: pd.DataFrame
    product_driver: pd.DataFrame
    alerts: pd.DataFrame
    transitions: dict[str, pd.DataFrame]
    cv_predictions: pd.DataFrame | None = None
    cv_metrics: pd.DataFrame | None = None
    rule_report: pd.DataFrame | None = None
    rule_confusion: pd.DataFrame | None = None
    ml_predictions: pd.DataFrame | None = None
    ml_metrics: pd.DataFrame | None = None
    ml_report: pd.DataFrame | None = None
    ml_confusion: pd.DataFrame | None = None
    ml_artifacts: MlArtifacts | None = None
    validation_error: str | None = None


class UiState:
    upload_path: Path | None = None
    result: AnalysisResult | None = None


state = UiState()
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def validate_workbook(path: Path) -> None:
    xl = pd.ExcelFile(path)
    sheets = set(xl.sheet_names)
    missing_sheets = REQUIRED_SHEETS.difference(sheets)
    if missing_sheets:
        raise ValueError(f"Workbook is missing required sheets: {sorted(missing_sheets)}")
    for sheet, required in REQUIRED_COLUMNS.items():
        columns = set(pd.read_excel(path, sheet_name=sheet, nrows=0).columns.astype(str))
        missing = required.difference(columns)
        if missing:
            raise ValueError(f"Sheet {sheet!r} is missing required columns: {sorted(missing)}")


def clean_ml_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = df.reindex(columns=ML_FEATURES).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for column in matrix.columns:
        if matrix[column].dtype == bool:
            matrix[column] = matrix[column].astype(int)
    return matrix.astype(float)


def select_time_folds(
    weekly_sales_full: pd.DataFrame,
    n_folds: int = 5,
    min_history_weeks: int = 60,
    horizon: int = 4,
) -> pd.DatetimeIndex:
    weeks = pd.DatetimeIndex(sorted(pd.to_datetime(weekly_sales_full["week_start"].dropna().unique())))
    usable = weeks[min_history_weeks : max(min_history_weeks, len(weeks) - horizon)]
    if len(usable) < n_folds:
        raise ValueError("Not enough weekly history for rolling validation.")
    positions = np.linspace(0, len(usable) - 1, n_folds).round().astype(int)
    return pd.DatetimeIndex(usable[positions])


def actual_future_labels(
    predicted_features: pd.DataFrame,
    weekly_sales_full: pd.DataFrame,
    anchor_week: pd.Timestamp,
    horizon: int,
    classification_config: dict[str, dict[str, float]],
) -> pd.DataFrame:
    future_weeks_actual = pd.date_range(anchor_week + pd.Timedelta(days=7), periods=horizon, freq=sd.WEEK_FREQ)
    actual = (
        weekly_sales_full[weekly_sales_full["week_start"].isin(future_weeks_actual)]
        .groupby(["client_id", "category_h", "analytical_block"], as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "actual_future_spend_4w"})
    )
    out = predicted_features.merge(actual, on=["client_id", "category_h", "analytical_block"], how="left")
    out["actual_future_spend_4w"] = out["actual_future_spend_4w"].fillna(0.0).clip(lower=0.0)
    actual_capture_spend = (out["spend_capture_window"] - out["spend_4w"] + out["actual_future_spend_4w"]).clip(lower=0.0)
    out["actual_future_capture_ratio_window"] = (
        actual_capture_spend / out["potential_capture_window"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=3.0)
    out["actual_future_trend_pct"] = sd.potential_scaled_change(
        out["actual_future_spend_4w"], out["spend_4w"], out["potential_4w"]
    )
    out["actual_classification_slope"] = out["client_weighted_forecast_product_slope"].where(
        out["client_weighted_forecast_product_slope"].abs() > sd.EPS,
        out["actual_future_trend_pct"],
    )
    out["actual_future_state"] = sd.classify_state(
        out["actual_future_spend_4w"],
        out["actual_future_capture_ratio_window"],
        out["actual_classification_slope"],
        out["analytical_block"],
        classification_config,
    )
    return out[out["classification_eligible"]].copy()


def run_pipeline_at_anchor(
    data: dict[str, pd.DataFrame],
    anchor_week: pd.Timestamp,
    forecast_weeks: int,
    model_mode: str,
    maxiter: int,
    model_config: dict[str, Any],
    classification_config: dict[str, dict[str, float]],
    client_window_config: dict[str, int],
) -> pd.DataFrame:
    sales = data["sales"]
    max_week = sd.week_start(pd.Series([anchor_week])).iloc[0]
    min_week = sales["week_start"].min()
    history_weeks = pd.date_range(min_week, max_week, freq=sd.WEEK_FREQ)
    future_weeks = pd.date_range(max_week + pd.Timedelta(days=7), periods=forecast_weeks, freq=sd.WEEK_FREQ)
    all_weeks = history_weeks.union(future_weeks)
    campaign_calendar = sd.build_campaign_calendar(data["campaigns"], all_weeks)
    weekly_all = sd.build_weekly_sales(sales, campaign_calendar)
    weekly_history = weekly_all[weekly_all["week_start"].isin(history_weeks)].copy()
    type_forecasts, factor_by_block = sd.forecast_type_sales(
        weekly_history, history_weeks, future_weeks, model_mode, maxiter, model_config
    )
    product_forecasts = sd.forecast_product_sales(
        weekly_history,
        data["products"],
        history_weeks,
        future_weeks,
        factor_by_block,
        model_mode,
        maxiter,
        model_config,
    )
    classifications, _ = sd.build_client_features(
        weekly_history,
        data["potential"],
        max_week,
        history_weeks,
        product_forecasts,
        classification_config,
        client_window_config,
    )
    return classifications


def evaluate_rolling_cv(
    data: dict[str, pd.DataFrame],
    weekly_sales_full: pd.DataFrame,
    folds: pd.DatetimeIndex,
    forecast_weeks: int,
    model_mode: str,
    maxiter: int,
    model_config: dict[str, Any],
    classification_config: dict[str, dict[str, float]],
    client_window_config: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_frames: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    for i, anchor in enumerate(folds, start=1):
        predicted = run_pipeline_at_anchor(
            data,
            anchor,
            forecast_weeks,
            model_mode,
            maxiter,
            model_config,
            classification_config,
            client_window_config,
        )
        evaluated = actual_future_labels(predicted, weekly_sales_full, anchor, forecast_weeks, classification_config)
        evaluated["fold"] = i
        evaluated["anchor_week"] = anchor
        y_true = evaluated["actual_future_state"]
        y_pred = evaluated["future_state"]
        fold_metrics.append(
            {
                "fold": i,
                "anchor_week": anchor,
                "rows": len(evaluated),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro"),
                "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            }
        )
        fold_frames.append(evaluated)
    return pd.concat(fold_frames, ignore_index=True), pd.DataFrame(fold_metrics)


def train_ml_candidate_from_cv(
    cv_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, MlArtifacts | None, pd.DataFrame | None]:
    rows = cv_predictions[cv_predictions["actual_future_state"].isin(STATE_LABELS)].copy()
    label_counts = rows["actual_future_state"].value_counts()
    valid_labels = label_counts[label_counts >= 5].index
    rows = rows[rows["actual_future_state"].isin(valid_labels)].copy()
    if rows["actual_future_state"].nunique() < 2:
        return None, None, None, None

    fold_results: list[dict[str, Any]] = []
    all_preds: list[pd.DataFrame] = []
    for test_fold in sorted(rows["fold"].unique()):
        train_df = rows[rows["fold"] != test_fold]
        test_df = rows[rows["fold"] == test_fold]
        if train_df["actual_future_state"].nunique() < 2 or test_df.empty:
            continue
        scaler = StandardScaler()
        encoder = LabelEncoder()
        x_train = scaler.fit_transform(clean_ml_matrix(train_df))
        y_train = encoder.fit_transform(train_df["actual_future_state"])
        x_test = scaler.transform(clean_ml_matrix(test_df))
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)
        clf.fit(x_train, y_train)
        pred = encoder.inverse_transform(clf.predict(x_test))
        part = test_df[["fold", "client_id", "category_h", "analytical_block", "actual_future_state", "future_state"]].copy()
        part["ml_future_state"] = pred
        all_preds.append(part)
        fold_results.append(
            {
                "fold": test_fold,
                "rows": len(test_df),
                "ml_accuracy": accuracy_score(test_df["actual_future_state"], pred),
                "ml_macro_f1": f1_score(test_df["actual_future_state"], pred, average="macro"),
                "ml_weighted_f1": f1_score(test_df["actual_future_state"], pred, average="weighted"),
            }
        )
    if not all_preds:
        return None, None, None, None

    pred_df = pd.concat(all_preds, ignore_index=True)
    metrics = pd.DataFrame(fold_results)
    final_scaler = StandardScaler()
    final_encoder = LabelEncoder()
    x_all = final_scaler.fit_transform(clean_ml_matrix(rows))
    y_all = final_encoder.fit_transform(rows["actual_future_state"])
    final_model = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)
    final_model.fit(x_all, y_all)
    coef = pd.DataFrame(final_model.coef_, columns=ML_FEATURES, index=final_encoder.classes_).T
    return pred_df, metrics, MlArtifacts(final_model, final_scaler, final_encoder, coef), rows


def report_dataframe(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, labels=STATE_LABELS, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})


def confusion_dataframe(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=STATE_LABELS)
    return pd.DataFrame(matrix, index=STATE_LABELS, columns=STATE_LABELS)


def run_full_analysis(
    input_path: Path,
    forecast_weeks: int,
    model_mode: str,
    cv_model_mode: str,
    run_validation: bool,
    maxiter: int,
    top_alerts: int,
) -> AnalysisResult:
    validate_workbook(input_path)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    classification_config = sd.load_classification_config(None)
    model_config = sd.load_model_config(None)
    client_window_config = sd.load_client_window_config(None)

    sheets = sd.load_workbook(input_path)
    data = sd.prepare_data(sheets)
    current_week, history_weeks, future_weeks, all_weeks = sd.build_week_calendar(data["sales"], None, forecast_weeks)
    campaigns = sd.build_campaign_calendar(data["campaigns"], all_weeks)
    weekly_all = sd.build_weekly_sales(data["sales"], campaigns)
    weekly_sales = weekly_all[weekly_all["week_start"].isin(history_weeks)].copy()

    type_forecasts, factor_by_block = sd.forecast_type_sales(
        weekly_sales, history_weeks, future_weeks, model_mode, maxiter, model_config
    )
    product_forecasts = sd.forecast_product_sales(
        weekly_sales,
        data["products"],
        history_weeks,
        future_weeks,
        factor_by_block,
        model_mode,
        maxiter,
        model_config,
    )
    classifications, product_driver = sd.build_client_features(
        weekly_sales,
        data["potential"],
        current_week,
        history_weeks,
        product_forecasts,
        classification_config,
        client_window_config,
    )
    alerts = sd.build_alerts(classifications, product_driver)
    transitions = sd.build_transition_tables(classifications)

    summary = {
        "input": input_path.name,
        "current_week": current_week,
        "forecast_weeks": forecast_weeks,
        "history_weeks": len(history_weeks),
        "sales_rows": len(data["sales"]),
        "weekly_sales_rows": len(weekly_sales),
        "clients_with_sales": int(data["sales"]["client_id"].nunique()),
        "products": int(data["products"]["product_id"].nunique()),
        "classification_rows": len(classifications),
        "classification_eligible_rows": int(classifications["classification_eligible"].sum()),
        "alerts": len(alerts),
        "high_priority_alerts": int((alerts.get("priority") == "High").sum()) if not alerts.empty else 0,
        "model_mode": model_mode,
        "cv_model_mode": cv_model_mode if run_validation else "not_run",
        "statsmodels_available": sd.HAS_STATSMODELS,
        "classification_config": classification_config,
        "model_config": model_config,
        "client_window_config": client_window_config,
    }

    result = AnalysisResult(
        run_dir=run_dir,
        summary=summary,
        weekly_sales=weekly_sales,
        type_forecasts=type_forecasts,
        product_forecasts=product_forecasts,
        classifications=classifications,
        product_driver=product_driver,
        alerts=alerts,
        transitions=transitions,
    )

    if run_validation:
        try:
            folds = select_time_folds(weekly_all, n_folds=5, min_history_weeks=60, horizon=forecast_weeks)
            cv_predictions, cv_metrics = evaluate_rolling_cv(
                data,
                weekly_all,
                folds,
                forecast_weeks,
                cv_model_mode,
                maxiter,
                model_config,
                classification_config,
                client_window_config,
            )
            result.cv_predictions = cv_predictions
            result.cv_metrics = cv_metrics
            result.rule_report = report_dataframe(cv_predictions["actual_future_state"], cv_predictions["future_state"])
            result.rule_confusion = confusion_dataframe(cv_predictions["actual_future_state"], cv_predictions["future_state"])
            ml_predictions, ml_metrics, ml_artifacts, _ = train_ml_candidate_from_cv(cv_predictions)
            result.ml_predictions = ml_predictions
            result.ml_metrics = ml_metrics
            result.ml_artifacts = ml_artifacts
            if ml_predictions is not None:
                result.ml_report = report_dataframe(ml_predictions["actual_future_state"], ml_predictions["ml_future_state"])
                result.ml_confusion = confusion_dataframe(ml_predictions["actual_future_state"], ml_predictions["ml_future_state"])
            summary["cv_rule_accuracy_mean"] = float(cv_metrics["accuracy"].mean())
            summary["cv_rule_macro_f1_mean"] = float(cv_metrics["macro_f1"].mean())
            summary["cv_ml_macro_f1_mean"] = None if ml_metrics is None else float(ml_metrics["ml_macro_f1"].mean())
        except Exception as exc:  # validation should not block the dashboard
            result.validation_error = str(exc)

    sd.write_outputs(run_dir, weekly_sales, type_forecasts, product_forecasts, classifications, alerts, summary, top_alerts, transitions)
    if result.cv_predictions is not None:
        result.cv_predictions.to_csv(run_dir / "rolling_cv_rule_predictions.csv", index=False)
    if result.cv_metrics is not None:
        result.cv_metrics.to_csv(run_dir / "rolling_cv_rule_metrics.csv", index=False)
    if result.ml_predictions is not None:
        result.ml_predictions.to_csv(run_dir / "rolling_cv_ml_predictions.csv", index=False)
    if result.ml_metrics is not None:
        result.ml_metrics.to_csv(run_dir / "rolling_cv_ml_metrics.csv", index=False)
    return result


def format_rows(df: pd.DataFrame, columns: list[str], limit: int = 500) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = df.loc[:, [col for col in columns if col in df.columns]].head(limit).copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].map(lambda x: None if pd.isna(x) else round(float(x), 4))
    return out.replace({np.nan: None}).to_dict("records")


def table_columns(columns: list[str]) -> list[dict[str, Any]]:
    return [{"name": col, "label": col.replace("_", " ").title(), "field": col, "sortable": True, "align": "left"} for col in columns]


def row_from_event(event: Any) -> dict[str, Any] | None:
    args = getattr(event, "args", None)
    if isinstance(args, dict):
        row = args.get("row", args)
        return row if isinstance(row, dict) else None
    if isinstance(args, (list, tuple)) and len(args) > 1 and isinstance(args[1], dict):
        return args[1]
    return None


def full_row_for_detail(result: AnalysisResult, row: dict[str, Any]) -> dict[str, Any]:
    client_id = str(row.get("client_id", ""))
    category = row.get("category_h")
    block = row.get("analytical_block")
    sources = [result.alerts, result.classifications]
    for source in sources:
        if source.empty:
            continue
        matched = source[
            (source["client_id"].astype(str) == client_id)
            & (source["category_h"] == category)
            & (source["analytical_block"] == block)
        ]
        if not matched.empty:
            full = matched.iloc[0].replace({np.nan: None}).to_dict()
            full.update({key: value for key, value in row.items() if value is not None})
            return full
    return row


def make_type_forecast_figure(result: AnalysisResult, last_year_only: bool = False) -> go.Figure:
    weekly = (
        result.weekly_sales.groupby(["analytical_block", "week_start"], as_index=False)["revenue"].sum()
    )
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    forecasts = result.type_forecasts.copy()
    forecasts["week_start"] = pd.to_datetime(forecasts["week_start"])
    if last_year_only and not weekly.empty:
        cutoff = weekly["week_start"].max() - pd.Timedelta(weeks=52)
        weekly = weekly[weekly["week_start"] >= cutoff]
    fig = go.Figure()
    for block in sorted(weekly["analytical_block"].dropna().unique()):
        hist = weekly[weekly["analytical_block"] == block]
        fut = forecasts[forecasts["analytical_block"] == block]
        fig.add_trace(go.Scatter(x=hist["week_start"], y=hist["revenue"], mode="lines", name=f"{block} history"))
        fig.add_trace(
            go.Scatter(
                x=fut["week_start"],
                y=fut["forecast_revenue"],
                mode="lines+markers",
                line={"dash": "dash"},
                name=f"{block} forecast",
            )
        )
    title = "Product Type Demand: Last 52 Weeks + 4 Week Forecast" if last_year_only else "Product Type Demand: Full History + 4 Week Forecast"
    fig.update_layout(title=title, xaxis_title="Week", yaxis_title="Revenue", template="plotly_white", hovermode="x unified")
    return fig


def make_confusion_figure(matrix: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(z=matrix.values, x=matrix.columns, y=matrix.index, colorscale="Blues", text=matrix.values, texttemplate="%{text}"))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual", template="plotly_white")
    return fig


def selected_product_ids(result: AnalysisResult, row: dict[str, Any]) -> list[str]:
    client_id = str(row.get("client_id", ""))
    category = row.get("category_h")
    block = row.get("analytical_block")
    driver = row.get("driver_product_id")
    subset = result.weekly_sales[
        (result.weekly_sales["client_id"].astype(str) == client_id)
        & (result.weekly_sales["category_h"] == category)
        & (result.weekly_sales["analytical_block"] == block)
    ].copy()
    if not subset.empty:
        cutoff = pd.to_datetime(subset["week_start"]).max() - pd.Timedelta(weeks=52)
        subset = subset[pd.to_datetime(subset["week_start"]) >= cutoff]
    top = subset.groupby("product_id")["revenue"].sum().sort_values(ascending=False).head(3).index.astype(str).tolist()
    ids = []
    if driver is not None and str(driver) not in {"", "None", "nan"}:
        ids.append(str(driver).replace(".0", ""))
    for product_id in top:
        if product_id not in ids:
            ids.append(product_id)
    return ids[:3]


def make_client_product_figure(result: AnalysisResult, row: dict[str, Any]) -> go.Figure:
    product_ids = selected_product_ids(result, row)
    client_id = str(row.get("client_id", ""))
    fig = go.Figure()
    for product_id in product_ids:
        hist = result.weekly_sales[
            (result.weekly_sales["client_id"].astype(str) == client_id)
            & (result.weekly_sales["product_id"].astype(str) == product_id)
        ]
        hist = hist.groupby("week_start", as_index=False)["revenue"].sum()
        hist["week_start"] = pd.to_datetime(hist["week_start"])
        if not hist.empty:
            hist = hist[hist["week_start"] >= hist["week_start"].max() - pd.Timedelta(weeks=52)]
        fut = result.product_forecasts[result.product_forecasts["product_id"].astype(str) == product_id].copy()
        fut["week_start"] = pd.to_datetime(fut["week_start"])
        if not fut.empty:
            last4 = float(hist.sort_values("week_start").tail(4)["revenue"].sum()) if not hist.empty else 0.0
            if last4 <= sd.EPS and not hist.empty:
                last4 = float(hist["revenue"].clip(lower=0).sum()) * 4.0 / max(hist["week_start"].nunique(), 1)
            factor = float(fut["product_forecast_factor"].iloc[0]) if "product_forecast_factor" in fut else 1.0
            forecast_total = max(0.0, last4 * factor)
            weights = fut["forecast_revenue"].clip(lower=0)
            weights = weights / weights.sum() if weights.sum() > sd.EPS else pd.Series(1 / len(fut), index=fut.index)
            fut["client_forecast_revenue"] = forecast_total * weights
        fig.add_trace(go.Scatter(x=hist["week_start"], y=hist["revenue"], mode="lines", name=f"Product {product_id} history"))
        if not fut.empty:
            fig.add_trace(
                go.Scatter(
                    x=fut["week_start"],
                    y=fut["client_forecast_revenue"],
                    mode="lines+markers",
                    line={"dash": "dash"},
                    name=f"Product {product_id} client signal",
                )
            )
    fig.update_layout(title=f"Client {client_id}: Top Product Signals", xaxis_title="Week", yaxis_title="Client revenue signal", template="plotly_white")
    return fig


def client_contributions(result: AnalysisResult, row: dict[str, Any]) -> pd.DataFrame:
    if result.ml_artifacts is None:
        return pd.DataFrame()
    row_df = pd.DataFrame([row])
    target = row.get("future_state")
    artifacts = result.ml_artifacts
    if target not in artifacts.encoder.classes_:
        target = artifacts.encoder.classes_[0]
    class_index = list(artifacts.encoder.classes_).index(target)
    scaled = artifacts.scaler.transform(clean_ml_matrix(row_df))[0]
    contributions = scaled * artifacts.model.coef_[class_index]
    return (
        pd.DataFrame({"feature": ML_FEATURES, "contribution": contributions, "standardized_value": scaled})
        .assign(abs_contribution=lambda d: d["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .head(12)
    )


def show_client_detail(row: dict[str, Any]) -> None:
    result = state.result
    if result is None:
        return
    row = full_row_for_detail(result, row)
    with ui.dialog() as dialog, ui.card().classes("w-full max-w-6xl"):
        ui.label(f"Client {row.get('client_id')} | {row.get('category_h')} | {row.get('current_state')} -> {row.get('future_state')}").classes("text-h5")
        ui.label(row.get("explanation", "No explanation available."))
        detail_cols = [
            "client_id",
            "category_h",
            "analytical_block",
            "annual_potential",
            "spend_4w",
            "predicted_spend_4w",
            "current_state",
            "future_state",
            "driver_product_id",
            "driver_forecast_product_slope_pct",
            "priority_score",
        ]
        ui.table(columns=table_columns(detail_cols), rows=format_rows(pd.DataFrame([row]), detail_cols, 1), pagination=1).classes("w-full")
        ui.plotly(make_client_product_figure(result, row)).classes("w-full h-96")
        if result.rule_report is not None:
            ui.label("Rule Classifier Validation").classes("text-h6")
            ui.table(columns=table_columns(result.rule_report.columns.tolist()), rows=format_rows(result.rule_report, result.rule_report.columns.tolist(), 20)).classes("w-full")
        if result.ml_artifacts is not None:
            future_state = row.get("future_state")
            ui.label("Logistic Regression Coefficients").classes("text-h6")
            coef = result.ml_artifacts.coefficients.copy()
            if future_state in coef.columns:
                coef_view = coef[[future_state]].assign(abs_coef=lambda d: d[future_state].abs()).sort_values("abs_coef", ascending=False).head(12).reset_index().rename(columns={"index": "feature"})
            else:
                coef_view = coef.abs().max(axis=1).sort_values(ascending=False).head(12).reset_index().rename(columns={"index": "feature", 0: "abs_coef"})
            ui.table(columns=table_columns(coef_view.columns.tolist()), rows=format_rows(coef_view, coef_view.columns.tolist(), 20)).classes("w-full")
            contrib = client_contributions(result, row)
            if not contrib.empty:
                ui.label("Client-Specific Logistic Contributions").classes("text-h6")
                ui.table(columns=table_columns(contrib.columns.tolist()), rows=format_rows(contrib, contrib.columns.tolist(), 20)).classes("w-full")
        elif result.validation_error:
            ui.label(f"Validation/ML details unavailable: {result.validation_error}").classes("text-negative")
        with ui.row().classes("justify-end w-full"):
            ui.button("Close", on_click=dialog.close)
    dialog.open()


summary_container: Any = None
plots_container: Any = None
validation_container: Any = None
alerts_container: Any = None


def render_metric(label: str, value: Any) -> None:
    with ui.card().classes("min-w-44"):
        ui.label(label).classes("text-caption text-grey-7")
        ui.label(str(value)).classes("text-h5")


def render_clickable_table(title: str, df: pd.DataFrame, columns: list[str], limit: int = 250) -> None:
    ui.label(title).classes("text-h5 q-mt-md")
    if df.empty:
        ui.label("No clients matched this condition.").classes("text-grey-7")
        return
    table = ui.table(columns=table_columns(columns), rows=format_rows(df, columns, limit), pagination=25).classes("w-full")
    table.props("dense flat bordered")
    table.on("rowClick", lambda e: show_client_detail(row_from_event(e) or {}))
    ui.label("Click a row to inspect the client, classifier evidence, and strongest product signals.").classes("text-caption text-grey-7")


def render_results() -> None:
    result = state.result
    if result is None:
        return
    summary_container.clear()
    plots_container.clear()
    validation_container.clear()
    alerts_container.clear()

    with summary_container:
        with ui.row().classes("w-full gap-3"):
            render_metric("Current week", pd.to_datetime(result.summary["current_week"]).date())
            render_metric("Clients", result.summary["clients_with_sales"])
            render_metric("Products", result.summary["products"])
            render_metric("Alerts", result.summary["alerts"])
            render_metric("High priority", result.summary["high_priority_alerts"])
            render_metric("Output folder", result.run_dir.name)

    with plots_container:
        ui.label("Demand Forecasts").classes("text-h4 q-mt-lg")
        ui.plotly(make_type_forecast_figure(result, last_year_only=False)).classes("w-full h-96")
        ui.plotly(make_type_forecast_figure(result, last_year_only=True)).classes("w-full h-96")

    with validation_container:
        ui.label("Validation").classes("text-h4 q-mt-lg")
        if result.validation_error:
            ui.label(f"Validation could not be completed: {result.validation_error}").classes("text-negative")
        if result.cv_metrics is not None:
            ui.label("Rule-Based Future State Validation").classes("text-h5")
            ui.table(columns=table_columns(result.cv_metrics.columns.tolist()), rows=format_rows(result.cv_metrics, result.cv_metrics.columns.tolist(), 10)).classes("w-full")
            if result.rule_confusion is not None:
                ui.plotly(make_confusion_figure(result.rule_confusion, "Rule Classifier Confusion Matrix")).classes("w-full h-96")
            if result.rule_report is not None:
                ui.table(columns=table_columns(result.rule_report.columns.tolist()), rows=format_rows(result.rule_report, result.rule_report.columns.tolist(), 20)).classes("w-full")
        if result.ml_metrics is not None:
            ui.label("Logistic Regression Candidate Validation").classes("text-h5 q-mt-md")
            ui.table(columns=table_columns(result.ml_metrics.columns.tolist()), rows=format_rows(result.ml_metrics, result.ml_metrics.columns.tolist(), 10)).classes("w-full")
            if result.ml_confusion is not None:
                ui.plotly(make_confusion_figure(result.ml_confusion, "Logistic Regression Confusion Matrix")).classes("w-full h-96")
            if result.ml_report is not None:
                ui.table(columns=table_columns(result.ml_report.columns.tolist()), rows=format_rows(result.ml_report, result.ml_report.columns.tolist(), 20)).classes("w-full")
            if result.ml_artifacts is not None:
                coef = result.ml_artifacts.coefficients.copy()
                for label in coef.columns:
                    coef_view = coef[[label]].assign(abs_coef=lambda d: d[label].abs()).sort_values("abs_coef", ascending=False).head(8).reset_index().rename(columns={"index": "feature"})
                    ui.label(f"Strongest standardized coefficients for {label}").classes("text-subtitle1 q-mt-sm")
                    ui.table(columns=table_columns(coef_view.columns.tolist()), rows=format_rows(coef_view, coef_view.columns.tolist(), 10)).classes("w-full")

    with alerts_container:
        ui.label("Client Alerts").classes("text-h4 q-mt-lg")
        alert_cols = [
            "client_id",
            "category_h",
            "analytical_block",
            "annual_potential",
            "current_state",
            "future_state",
            "priority",
            "priority_score",
            "spend_4w",
            "predicted_spend_4w",
            "driver_product_id",
            "driver_forecast_product_slope_pct",
        ]
        alerts = result.alerts.copy()
        to_risky = alerts[(alerts["future_state"] == "Risky") & (alerts["current_state"] != "Risky")].sort_values("annual_potential", ascending=False)
        loyal_to_promiscuous = alerts[(alerts["current_state"] == "Loyal") & (alerts["future_state"] == "Promiscuous")].sort_values("annual_potential", ascending=False)
        promising = result.classifications[(result.classifications["future_state"] == "Promising") & (result.classifications["classification_eligible"])].sort_values("annual_potential", ascending=False)
        render_clickable_table("Any State -> Risky", to_risky, alert_cols)
        render_clickable_table("Loyal -> Promiscuous", loyal_to_promiscuous, alert_cols)
        render_clickable_table("Promising Clients", promising, [col for col in alert_cols if col in promising.columns])


async def handle_upload(event: Any) -> None:
    file_name = Path(event.name).name
    if not file_name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        ui.notify("Upload an Excel workbook (.xlsx, .xlsm, .xls).", type="negative")
        return
    destination = DATA_DIR / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{file_name}"
    with destination.open("wb") as out:
        shutil.copyfileobj(event.content, out)
    state.upload_path = destination
    ui.notify(f"Uploaded {file_name}", type="positive")
    upload_label.set_text(f"Selected file: {destination.name}")


async def run_analysis() -> None:
    if state.upload_path is None:
        ui.notify("Upload a workbook first.", type="warning")
        return
    run_button.disable()
    status_label.set_text("Running pipeline. This can take several minutes in SARIMA/SARIMAX auto mode...")
    try:
        result = await run.io_bound(
            run_full_analysis,
            state.upload_path,
            int(forecast_weeks_input.value),
            model_mode_select.value,
            cv_mode_select.value,
            bool(validation_checkbox.value),
            int(maxiter_input.value),
            int(top_alerts_input.value),
        )
        state.result = result
        status_label.set_text(f"Analysis complete. Outputs written to {result.run_dir}.")
        render_results()
        ui.notify("Analysis complete", type="positive")
    except Exception as exc:
        status_label.set_text(f"Analysis failed: {exc}")
        ui.notify(str(exc), type="negative", multi_line=True)
    finally:
        run_button.enable()


ui.colors(primary="#124c5f", secondary="#1f7a8c", accent="#ffb703", positive="#2a9d8f")

with ui.header().classes("items-center justify-between"):
    ui.label("Smart Demand Signals").classes("text-h5")
    ui.label("Hackathon 2026 | NiceGUI Dashboard").classes("text-subtitle2")

with ui.column().classes("w-full max-w-7xl mx-auto q-pa-md gap-4"):
    with ui.card().classes("w-full"):
        ui.label("Upload and Run").classes("text-h4")
        ui.label("Upload an Excel workbook with Ventas, Productos, Clientes, Potencial, and Campañas sheets.").classes("text-grey-7")
        ui.upload(on_upload=handle_upload, auto_upload=True, max_file_size=200_000_000).props("accept=.xlsx,.xlsm,.xls").classes("w-full")
        upload_label = ui.label("No file selected.").classes("text-caption")
        with ui.row().classes("items-end gap-4"):
            model_mode_select = ui.select(["auto", "fallback"], value="auto", label="Forecast mode")
            cv_mode_select = ui.select(["fallback", "auto"], value="fallback", label="Validation mode")
            validation_checkbox = ui.checkbox("Run rolling validation", value=True)
            forecast_weeks_input = ui.number("Forecast weeks", value=4, min=1, max=12, step=1)
            maxiter_input = ui.number("SARIMA maxiter", value=60, min=10, max=300, step=10)
            top_alerts_input = ui.number("Top alerts exported", value=250, min=50, max=5000, step=50)
        run_button = ui.button("Run Analysis", on_click=run_analysis).props("unelevated")
        status_label = ui.label("Ready.").classes("text-grey-8")

    summary_container = ui.column().classes("w-full")
    plots_container = ui.column().classes("w-full")
    validation_container = ui.column().classes("w-full")
    alerts_container = ui.column().classes("w-full")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


ui.run(
    host="0.0.0.0",
    port=APP_PORT,
    title="Smart Demand Signals",
    storage_secret=os.getenv("NICEGUI_STORAGE_SECRET", "change-this-secret"),
    reload=False,
)
