#!/usr/bin/env python3
"""Smart Demand Signals pipeline.

This script builds a weekly analytical dataset from the hackathon workbook,
fits a hierarchical demand forecast, classifies clients over the last 4 weeks,
projects their classification 4 weeks ahead, and generates prioritized alerts.

Run:
    python project.py --input Datasets.xlsx --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - user environment guard
    raise SystemExit(
        "Missing required packages. Install them with: "
        "python -m pip install -r requirements.txt"
    ) from exc


try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover - optional fallback path
    SARIMAX = None
    HAS_STATSMODELS = False


WEEK_FREQ = "W-MON"
EPS = 1e-9


DEFAULT_CLASSIFICATION_CONFIG = {
    "Commodities": {
        "potential_achieved_threshold": 0.60,
        "slope_threshold": 0.15,
    },
    "Productos Técnicos": {
        "potential_achieved_threshold": 0.45,
        "slope_threshold": 0.30,
    },
    "default": {
        "potential_achieved_threshold": 0.55,
        "slope_threshold": 0.20,
    },
}

STATE_ORDER = ["Loyal", "Promising", "Risky", "Promiscuous"]

DEFAULT_MODEL_CONFIG = {
    "type_order": (1, 1, 1),
    "type_seasonal_order": (1, 0, 0, 52),
    "product_order": (1, 0, 1),
    "product_seasonal_order": (0, 0, 0, 0),
    "min_model_weeks": 36,
    "min_seasonal_weeks": 104,
}


@dataclass
class ForecastResult:
    forecast: pd.Series
    method: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Smart Demand Signals outputs.")
    parser.add_argument("--input", default="Datasets.xlsx", help="Path to the Excel workbook.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated files.")
    parser.add_argument("--forecast-weeks", type=int, default=4, help="Weeks to forecast ahead.")
    parser.add_argument(
        "--history-weeks",
        type=int,
        default=0,
        help="Optional cap on forecasting history. Use 0 to keep all available history.",
    )
    parser.add_argument(
        "--current-end",
        default=None,
        help="Optional analysis end date, e.g. 2025-12-29. Defaults to the last completed sales week.",
    )
    parser.add_argument(
        "--model-mode",
        choices=["auto", "fallback"],
        default="auto",
        help="Use SARIMA/SARIMAX when available, or force fallback forecasts.",
    )
    parser.add_argument("--maxiter", type=int, default=60, help="Max optimizer iterations per time-series model.")
    parser.add_argument("--top-alerts", type=int, default=250, help="Rows kept in top_alerts.csv.")
    parser.add_argument(
        "--classification-config-json",
        default=None,
        help="Optional JSON file overriding block potential/slope thresholds for the four-state classifier.",
    )
    parser.add_argument(
        "--model-config-json",
        default=None,
        help="Optional JSON file overriding SARIMA/SARIMAX orders and minimum history settings.",
    )
    return parser.parse_args()


def load_classification_config(path: str | None = None) -> dict[str, dict[str, float]]:
    config = {block: values.copy() for block, values in DEFAULT_CLASSIFICATION_CONFIG.items()}
    if path:
        overrides = json.loads(Path(path).read_text(encoding="utf-8"))
        for block, values in overrides.items():
            config.setdefault(block, config["default"].copy()).update(values)
    return config


def load_model_config(path: str | None = None) -> dict[str, Any]:
    config = DEFAULT_MODEL_CONFIG.copy()
    if path:
        overrides = json.loads(Path(path).read_text(encoding="utf-8"))
        config.update(overrides)
    for key in ["type_order", "type_seasonal_order", "product_order", "product_seasonal_order"]:
        config[key] = tuple(config[key])
    return config


def normalize_id(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = numeric.round().astype("Int64").astype("string")
    fallback = series.astype("string").str.strip()
    out = out.mask(numeric.isna(), fallback)
    return out.fillna("").str.replace(r"\.0$", "", regex=True)


def excel_or_datetime(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_dates = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    parsed_dates = pd.to_datetime(series, errors="coerce")
    return numeric_dates.fillna(parsed_dates)


def week_start(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series)
    return (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.normalize()


def clean_money(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def load_workbook(path: Path) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None)
    required = {"Ventas", "Productos", "Clientes", "Potencial", "Campañas"}
    missing = required.difference(sheets)
    if missing:
        raise ValueError(f"Workbook is missing required sheets: {sorted(missing)}")
    return sheets


def prepare_data(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    products = sheets["Productos"].rename(
        columns={
            "Id.Prod": "product_id",
            "Bloque analítico": "analytical_block",
            "Categoria_H": "category_h",
            "Familia_H": "family_h",
        }
    )
    products["product_id"] = normalize_id(products["product_id"])
    products = products.dropna(subset=["product_id"]).drop_duplicates("product_id")

    clients = sheets["Clientes"].rename(
        columns={"Id. Cliente": "client_id", "11": "postal_code", 11: "postal_code", "Provincia": "province"}
    )
    if "province" not in clients.columns and len(clients.columns) >= 3:
        clients = clients.rename(columns={clients.columns[2]: "province"})
    if "province" not in clients.columns:
        clients["province"] = "Unknown"
    clients["client_id"] = normalize_id(clients["client_id"])
    clients = clients.dropna(subset=["client_id"]).drop_duplicates("client_id")

    potential = sheets["Potencial"].rename(
        columns={
            "Id.Cliente": "client_id",
            "Familia": "potential_family",
            "Categoria Productos": "category_h",
            "Potencial_H": "annual_potential",
        }
    )
    potential["client_id"] = normalize_id(potential["client_id"])
    potential["annual_potential"] = clean_money(potential["annual_potential"])
    potential = (
        potential.groupby(["client_id", "category_h", "potential_family"], as_index=False)["annual_potential"]
        .sum()
        .dropna(subset=["client_id", "category_h"])
    )

    campaigns = sheets["Campañas"].rename(
        columns={"Campaña": "campaign", "Fecha inicio": "start_date", "Fecha fin": "end_date"}
    )
    campaigns["start_date"] = excel_or_datetime(campaigns["start_date"])
    campaigns["end_date"] = excel_or_datetime(campaigns["end_date"])
    campaigns = campaigns.dropna(subset=["campaign", "start_date", "end_date"])

    sales = sheets["Ventas"].rename(
        columns={
            "Num.Fact": "invoice_id",
            "Fecha": "date",
            "Id. Cliente": "client_id",
            "Id. Producto": "product_id",
            "Unidades": "units",
            "Valores_H": "revenue",
        }
    )
    sales["date"] = excel_or_datetime(sales["date"])
    sales["client_id"] = normalize_id(sales["client_id"])
    sales["product_id"] = normalize_id(sales["product_id"])
    sales["units"] = clean_money(sales["units"])
    sales["revenue"] = clean_money(sales["revenue"])
    sales = sales.dropna(subset=["date", "client_id", "product_id"])
    sales["week_start"] = week_start(sales["date"])
    sales = sales.merge(products, on="product_id", how="left", validate="many_to_one")
    sales = sales.merge(clients[["client_id", "province"]], on="client_id", how="left")
    sales["net_revenue"] = sales["revenue"]
    sales["gross_revenue"] = sales["revenue"].clip(lower=0)
    sales["is_return"] = sales["revenue"] < 0

    category_block = (
        products.groupby("category_h", as_index=False)["analytical_block"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    )
    potential = potential.merge(category_block, on="category_h", how="left")

    return {
        "sales": sales,
        "products": products,
        "clients": clients,
        "potential": potential,
        "campaigns": campaigns,
    }


def build_week_calendar(sales: pd.DataFrame, current_end: str | None, forecast_weeks: int) -> tuple[pd.Timestamp, pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    max_date = pd.to_datetime(current_end) if current_end else sales["date"].max()
    current_week = week_start(pd.Series([max_date])).iloc[0]
    if current_end is None and max_date.weekday() != 6:
        current_week = current_week - pd.Timedelta(days=7)
    min_week = sales["week_start"].min()
    history_weeks = pd.date_range(min_week, current_week, freq=WEEK_FREQ)
    future_weeks = pd.date_range(current_week + pd.Timedelta(days=7), periods=forecast_weeks, freq=WEEK_FREQ)
    all_weeks = history_weeks.union(future_weeks)
    return current_week, history_weeks, future_weeks, all_weeks


def apply_history_limit(history_weeks: pd.DatetimeIndex, history_limit: int | None) -> pd.DatetimeIndex:
    if history_limit is None or history_limit <= 0 or history_limit >= len(history_weeks):
        return history_weeks
    minimum_for_models = max(history_limit, 60)
    return pd.DatetimeIndex(history_weeks[-minimum_for_models:])


def build_campaign_calendar(campaigns: pd.DataFrame, all_weeks: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for week in all_weeks:
        week_end = week + pd.Timedelta(days=6)
        active = campaigns[(campaigns["start_date"] <= week_end) & (campaigns["end_date"] >= week)]
        rows.append(
            {
                "week_start": week,
                "campaign_active": int(not active.empty),
                "campaign_count": int(active["campaign"].nunique()),
                "campaigns": ",".join(active["campaign"].astype(str).unique()[:5]),
            }
        )
    return pd.DataFrame(rows)


def build_weekly_sales(sales: pd.DataFrame, campaign_calendar: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        sales.groupby(
            [
                "week_start",
                "client_id",
                "product_id",
                "analytical_block",
                "category_h",
                "family_h",
                "province",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            revenue=("net_revenue", "sum"),
            gross_revenue=("gross_revenue", "sum"),
            units=("units", "sum"),
            num_orders=("invoice_id", "nunique"),
            return_rows=("is_return", "sum"),
        )
        .merge(campaign_calendar, on="week_start", how="left")
    )
    grouped["campaign_active"] = grouped["campaign_active"].fillna(0).astype(int)
    grouped["campaign_count"] = grouped["campaign_count"].fillna(0).astype(int)
    return grouped


def fallback_forecast(y: pd.Series, future_index: pd.DatetimeIndex) -> ForecastResult:
    y = y.astype(float).fillna(0.0)
    preds: list[float] = []
    recent_mean = max(0.0, float(y.tail(8).mean())) if len(y) else 0.0
    for i in range(len(future_index)):
        if len(y) >= 56:
            seasonal_position = -52 + i
            value = float(y.iloc[seasonal_position]) if abs(seasonal_position) <= len(y) else recent_mean
            preds.append(max(0.0, value))
        else:
            preds.append(recent_mean)
    return ForecastResult(pd.Series(preds, index=future_index), "fallback_seasonal_or_rolling_mean")


def model_forecast(
    y: pd.Series,
    future_index: pd.DatetimeIndex,
    mode: str,
    maxiter: int,
    model_config: dict[str, Any] | None = None,
    exog: pd.DataFrame | None = None,
    future_exog: pd.DataFrame | None = None,
    product_level: bool = False,
) -> ForecastResult:
    model_config = model_config or DEFAULT_MODEL_CONFIG
    y = y.astype(float).fillna(0.0)
    if mode == "fallback" or not HAS_STATSMODELS or len(y) < int(model_config["min_model_weeks"]) or y.abs().sum() <= EPS:
        return fallback_forecast(y, future_index)

    order = model_config["product_order"] if product_level else model_config["type_order"]
    configured_seasonal = model_config["product_seasonal_order"] if product_level else model_config["type_seasonal_order"]
    seasonal_order = (0, 0, 0, 0) if len(y) < int(model_config["min_seasonal_weeks"]) else configured_seasonal

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                exog=exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=maxiter)
            predicted = fitted.get_forecast(steps=len(future_index), exog=future_exog).predicted_mean
        predicted = pd.Series(np.maximum(predicted.to_numpy(dtype=float), 0.0), index=future_index)
        method = "SARIMAX" if exog is not None else "SARIMA"
        return ForecastResult(predicted, method)
    except Exception as exc:  # pragma: no cover - model robustness path
        fallback = fallback_forecast(y, future_index)
        fallback.error = str(exc)[:300]
        return fallback


def demand_factor(series: pd.Series) -> pd.Series:
    baseline = series.rolling(12, min_periods=4).mean().replace(0, np.nan)
    factor = (series / baseline).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return factor.clip(lower=0.25, upper=3.0)


def forecast_type_sales(
    weekly_sales: pd.DataFrame,
    history_weeks: pd.DatetimeIndex,
    future_weeks: pd.DatetimeIndex,
    mode: str,
    maxiter: int,
    model_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    model_config = model_config or DEFAULT_MODEL_CONFIG
    block_weekly = (
        weekly_sales.groupby(["analytical_block", "week_start"], as_index=False)["revenue"].sum()
    )
    rows: list[pd.DataFrame] = []
    factor_by_block: dict[str, pd.Series] = {}
    for block in sorted(block_weekly["analytical_block"].dropna().unique()):
        y = (
            block_weekly.loc[block_weekly["analytical_block"] == block]
            .set_index("week_start")["revenue"]
            .reindex(history_weeks, fill_value=0.0)
        )
        result = model_forecast(y, future_weeks, mode=mode, maxiter=maxiter, model_config=model_config)
        history_factor = demand_factor(y)
        future_factor = (result.forecast / max(float(y.tail(12).mean()), EPS)).clip(lower=0.25, upper=3.0)
        factor_by_block[block] = pd.concat([history_factor, future_factor])
        part = pd.DataFrame(
            {
                "analytical_block": block,
                "week_start": future_weeks,
                "forecast_revenue": result.forecast.to_numpy(),
                "recent_4w_revenue": float(y.tail(4).sum()),
                "forecast_4w_revenue": float(result.forecast.sum()),
                "forecast_change_pct": safe_pct_change(float(result.forecast.sum()), float(y.tail(4).sum())),
                "model_method": result.method,
                "model_error": result.error,
            }
        )
        rows.append(part)
    return pd.concat(rows, ignore_index=True), factor_by_block


def forecast_product_sales(
    weekly_sales: pd.DataFrame,
    products: pd.DataFrame,
    history_weeks: pd.DatetimeIndex,
    future_weeks: pd.DatetimeIndex,
    factor_by_block: dict[str, pd.Series],
    mode: str,
    maxiter: int,
    model_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    model_config = model_config or DEFAULT_MODEL_CONFIG
    product_weekly = weekly_sales.groupby(["product_id", "week_start"], as_index=False)["revenue"].sum()
    rows: list[pd.DataFrame] = []
    product_meta = products.set_index("product_id")[["analytical_block", "category_h", "family_h"]].to_dict("index")

    for product_id, meta in product_meta.items():
        y = (
            product_weekly.loc[product_weekly["product_id"] == product_id]
            .set_index("week_start")["revenue"]
            .reindex(history_weeks, fill_value=0.0)
        )
        block = meta.get("analytical_block")
        block_factor = factor_by_block.get(block, pd.Series(1.0, index=history_weeks.union(future_weeks)))
        exog = pd.DataFrame({"type_demand_factor": block_factor.reindex(history_weeks).fillna(1.0)}, index=history_weeks)
        future_exog = pd.DataFrame({"type_demand_factor": block_factor.reindex(future_weeks).fillna(1.0)}, index=future_weeks)
        result = model_forecast(
            y,
            future_weeks,
            mode=mode,
            maxiter=maxiter,
            model_config=model_config,
            exog=exog,
            future_exog=future_exog,
            product_level=True,
        )
        current_4w = float(y.tail(4).sum())
        previous_4w = float(y.iloc[-8:-4].sum()) if len(y) >= 8 else 0.0
        forecast_4w = float(result.forecast.sum())
        fallback_scale = float(y.tail(12).mean()) * 4 if len(y) else 1.0
        factor = forecast_4w / max(current_4w, fallback_scale, EPS)
        historical_slope_pct = safe_scaled_change(current_4w, previous_4w, fallback_scale)
        forecast_slope_pct = safe_scaled_change(forecast_4w, current_4w, fallback_scale)
        part = pd.DataFrame(
            {
                "product_id": product_id,
                "analytical_block": block,
                "category_h": meta.get("category_h"),
                "family_h": meta.get("family_h"),
                "week_start": future_weeks,
                "forecast_revenue": result.forecast.to_numpy(),
                "recent_4w_revenue": current_4w,
                "previous_4w_revenue": previous_4w,
                "forecast_4w_revenue": forecast_4w,
                "forecast_change_pct": safe_pct_change(forecast_4w, current_4w),
                "historical_product_slope_pct": historical_slope_pct,
                "forecast_product_slope_pct": forecast_slope_pct,
                "product_forecast_factor": float(np.clip(factor, 0.2, 2.5)),
                "model_method": result.method,
                "model_error": result.error,
            }
        )
        rows.append(part)

    return pd.concat(rows, ignore_index=True)


def safe_pct_change(new: float, old: float) -> float:
    if abs(old) <= EPS:
        return 0.0 if abs(new) <= EPS else 1.0
    return (new - old) / abs(old)


def safe_scaled_change(new: float, old: float, fallback_scale: float = 1.0) -> float:
    denominator = max(abs(old), abs(fallback_scale), 1.0)
    return (new - old) / denominator


def potential_scaled_change(new: pd.Series, old: pd.Series, potential_4w: pd.Series) -> pd.Series:
    denominator = pd.concat(
        [old.abs(), (potential_4w * 0.10).abs(), pd.Series(1.0, index=old.index)],
        axis=1,
    ).max(axis=1)
    return ((new - old) / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def sum_window(df: pd.DataFrame, weeks: pd.DatetimeIndex, keys: list[str], value_col: str = "revenue") -> pd.DataFrame:
    subset = df[df["week_start"].isin(weeks)]
    if subset.empty:
        return pd.DataFrame(columns=keys + [value_col])
    return subset.groupby(keys, as_index=False)[value_col].sum()


def build_client_features(
    weekly_sales: pd.DataFrame,
    potential: pd.DataFrame,
    current_week: pd.Timestamp,
    history_weeks: pd.DatetimeIndex,
    product_forecasts: pd.DataFrame,
    classification_config: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    classification_config = classification_config or DEFAULT_CLASSIFICATION_CONFIG
    current_4 = pd.DatetimeIndex(history_weeks[-4:])
    prev_4 = pd.DatetimeIndex(history_weeks[-8:-4]) if len(history_weeks) >= 8 else pd.DatetimeIndex([])
    current_12 = pd.DatetimeIndex(history_weeks[-12:])
    current_52 = pd.DatetimeIndex(history_weeks[-52:])

    cat_sales = (
        weekly_sales.groupby(["client_id", "category_h", "analytical_block", "week_start"], dropna=False, as_index=False)
        .agg(revenue=("revenue", "sum"), num_orders=("num_orders", "sum"))
    )
    base_from_potential = potential[["client_id", "category_h", "potential_family", "analytical_block", "annual_potential"]]
    base_from_sales = cat_sales[["client_id", "category_h", "analytical_block"]].drop_duplicates()
    base = base_from_potential.merge(base_from_sales, on=["client_id", "category_h", "analytical_block"], how="outer")
    base["annual_potential"] = base["annual_potential"].fillna(0.0)
    base["potential_family"] = base["potential_family"].fillna("Unknown")
    base = base.drop_duplicates(["client_id", "category_h", "analytical_block", "potential_family"])

    spend_4 = sum_window(cat_sales, current_4, ["client_id", "category_h", "analytical_block"]).rename(columns={"revenue": "spend_4w"})
    prev_spend_4 = sum_window(cat_sales, prev_4, ["client_id", "category_h", "analytical_block"]).rename(columns={"revenue": "prev_spend_4w"})
    spend_12 = sum_window(cat_sales, current_12, ["client_id", "category_h", "analytical_block"]).rename(columns={"revenue": "spend_12w"})
    spend_52 = sum_window(cat_sales, current_52, ["client_id", "category_h", "analytical_block"]).rename(columns={"revenue": "spend_52w"})

    active = cat_sales[cat_sales["week_start"].isin(current_12) & (cat_sales["revenue"] > 0)]
    active_12 = active.groupby(["client_id", "category_h", "analytical_block"], as_index=False)["week_start"].nunique().rename(columns={"week_start": "active_weeks_12w"})
    active_4 = active[active["week_start"].isin(current_4)].groupby(["client_id", "category_h", "analytical_block"], as_index=False)["week_start"].nunique().rename(columns={"week_start": "active_weeks_4w"})
    active_52 = (
        cat_sales[cat_sales["week_start"].isin(current_52) & (cat_sales["revenue"] > 0)]
        .groupby(["client_id", "category_h", "analytical_block"], as_index=False)["week_start"]
        .nunique()
        .rename(columns={"week_start": "active_weeks_52w"})
    )
    weekly_stats = (
        cat_sales[cat_sales["week_start"].isin(current_12)]
        .groupby(["client_id", "category_h", "analytical_block"], as_index=False)["revenue"]
        .agg(weekly_mean_12w="mean", weekly_std_12w="std")
    )
    last_purchase = (
        cat_sales[(cat_sales["week_start"] <= current_week) & (cat_sales["revenue"] > 0)]
        .groupby(["client_id", "category_h", "analytical_block"], as_index=False)["week_start"]
        .max()
        .rename(columns={"week_start": "last_purchase_week"})
    )

    features = base.merge(spend_4, on=["client_id", "category_h", "analytical_block"], how="left")
    features = features.merge(prev_spend_4, on=["client_id", "category_h", "analytical_block"], how="left")
    features = features.merge(spend_12, on=["client_id", "category_h", "analytical_block"], how="left")
    features = features.merge(spend_52, on=["client_id", "category_h", "analytical_block"], how="left")
    features = features.merge(active_12, on=["client_id", "category_h", "analytical_block"], how="left")
    features = features.merge(active_4, on=["client_id", "category_h", "analytical_block"], how="left")
    features = features.merge(active_52, on=["client_id", "category_h", "analytical_block"], how="left")
    features = features.merge(weekly_stats, on=["client_id", "category_h", "analytical_block"], how="left")
    features = features.merge(last_purchase, on=["client_id", "category_h", "analytical_block"], how="left")

    fill_zero = [
        "spend_4w",
        "prev_spend_4w",
        "spend_12w",
        "spend_52w",
        "active_weeks_12w",
        "active_weeks_4w",
        "active_weeks_52w",
        "weekly_mean_12w",
        "weekly_std_12w",
    ]
    features[fill_zero] = features[fill_zero].fillna(0.0)
    features["potential_4w"] = features["annual_potential"] * 4.0 / 52.0
    features["capture_ratio"] = features["spend_4w"] / features["potential_4w"].replace(0, np.nan)
    features["capture_ratio"] = features["capture_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=3.0)
    features["trend_4w_vs_prev_4w"] = [safe_pct_change(n, o) for n, o in zip(features["spend_4w"], features["prev_spend_4w"])]
    features["volatility_12w"] = features["weekly_std_12w"] / features["weekly_mean_12w"].replace(0, np.nan)
    features["volatility_12w"] = features["volatility_12w"].replace([np.inf, -np.inf], np.nan).fillna(2.0).clip(lower=0.0, upper=3.0)
    features["weeks_since_last_purchase"] = ((current_week - features["last_purchase_week"]).dt.days / 7.0).fillna(999.0)
    features["regularity_score"] = regularity_score(features)
    features["relationship_status"] = relationship_status(features)
    features["classification_eligible"] = features["relationship_status"].ne("No Signal")
    features["abnormal_silence"] = abnormal_silence(features)
    features["capture_strength"] = contextual_strength(features, "capture_ratio", "spend_4w")
    features["observed_trend_pct"] = potential_scaled_change(features["spend_4w"], features["prev_spend_4w"], features["potential_4w"])

    features, product_driver = add_future_projection(features, weekly_sales, product_forecasts)
    features["classification_observed_slope"] = features["client_weighted_historical_product_slope"].where(
        features["client_weighted_historical_product_slope"].abs() > EPS,
        features["observed_trend_pct"],
    )
    features["current_state"] = classify_state(
        features["spend_4w"],
        features["capture_ratio"],
        features["classification_observed_slope"],
        features["analytical_block"],
        classification_config,
    )
    features["predicted_capture_ratio"] = features["predicted_spend_4w"] / features["potential_4w"].replace(0, np.nan)
    features["predicted_capture_ratio"] = features["predicted_capture_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=3.0)
    features["predicted_capture_strength"] = contextual_strength(features, "predicted_capture_ratio", "predicted_spend_4w")
    features["forecast_trend_pct"] = potential_scaled_change(features["predicted_spend_4w"], features["spend_4w"], features["potential_4w"])
    features["classification_forecast_slope"] = features["client_weighted_forecast_product_slope"].where(
        features["client_weighted_forecast_product_slope"].abs() > EPS,
        features["forecast_trend_pct"],
    )
    features["future_state"] = classify_state(
        features["predicted_spend_4w"],
        features["predicted_capture_ratio"],
        features["classification_forecast_slope"],
        features["analytical_block"],
        classification_config,
    )
    return features, product_driver


def regularity_score(features: pd.DataFrame) -> pd.Series:
    expected_active = np.where(features["analytical_block"].eq("Commodities"), 2.0, 3.0)
    active_component = np.minimum(features["active_weeks_12w"].to_numpy(dtype=float) / expected_active, 1.0)
    volatility_component = 1.0 - np.minimum(features["volatility_12w"].to_numpy(dtype=float), 2.0) / 2.0
    score = 0.7 * active_component + 0.3 * volatility_component
    return pd.Series(np.clip(score, 0.0, 1.0), index=features.index)


def relationship_status(features: pd.DataFrame) -> pd.Series:
    active = (features["spend_4w"] > 0) | (features["active_weeks_4w"] > 0)
    is_commodity = features["analytical_block"].eq("Commodities")
    recent_window = np.where(is_commodity, features["active_weeks_12w"] > 0, features["active_weeks_52w"] >= 2)
    dormant_window = np.where(
        is_commodity,
        (features["active_weeks_52w"] > 0) & (features["active_weeks_12w"] == 0),
        (features["active_weeks_52w"] == 1),
    )
    recently_inactive = (~active) & recent_window
    dormant = (~active) & (~recently_inactive) & dormant_window
    prospect = (~active) & (~recently_inactive) & (~dormant) & (features["annual_potential"] > 0)
    status = np.full(len(features), "No Signal", dtype=object)
    status[prospect.to_numpy()] = "Prospect"
    status[dormant.to_numpy()] = "Dormant relationship"
    status[recently_inactive.to_numpy()] = "Recently inactive"
    status[active.to_numpy()] = "Active"
    return pd.Series(status, index=features.index)


def abnormal_silence(features: pd.DataFrame) -> pd.Series:
    is_commodity = features["analytical_block"].eq("Commodities")
    commodity_silence = is_commodity & (features["weeks_since_last_purchase"] >= 8) & (features["active_weeks_52w"] >= 3)
    technical_silence = (~is_commodity) & (features["weeks_since_last_purchase"] >= 16) & (features["active_weeks_52w"] >= 3)
    recently_regular = features["regularity_score"] >= np.where(is_commodity, 0.35, 0.30)
    return (commodity_silence | technical_silence) & recently_regular


def contextual_strength(features: pd.DataFrame, ratio_col: str, spend_col: str) -> pd.Series:
    active = features["relationship_status"].eq("Active") & (features[spend_col] > 0)
    strength = pd.Series(0.0, index=features.index)
    group_cols = ["analytical_block", "category_h"]
    for _, idx in features[active].groupby(group_cols).groups.items():
        values = features.loc[idx, ratio_col].clip(lower=0, upper=3)
        ranks = values.rank(pct=True, method="average")
        strength.loc[idx] = ranks.fillna(0.0)
    absolute_bonus = (features[ratio_col] / 0.5).clip(lower=0.0, upper=1.0)
    return pd.concat([strength, absolute_bonus], axis=1).max(axis=1).fillna(0.0)


def classify_state(
    spend_4w: pd.Series,
    capture_ratio: pd.Series,
    trend_pct: pd.Series,
    analytical_block: pd.Series,
    classification_config: dict[str, dict[str, float]] | None = None,
) -> pd.Series:
    classification_config = classification_config or DEFAULT_CLASSIFICATION_CONFIG
    state = np.full(len(spend_4w), "Promiscuous", dtype=object)
    capture = capture_ratio.fillna(0.0).to_numpy(dtype=float)
    trend = trend_pct.fillna(0.0).to_numpy(dtype=float)

    for block in analytical_block.fillna("default").unique():
        mask = analytical_block.fillna("default").eq(block).to_numpy()
        config = classification_config.get(block, classification_config["default"])
        potential_threshold = config["potential_achieved_threshold"]
        slope_threshold = config["slope_threshold"]
        loyal = capture >= potential_threshold
        risky = trend <= -slope_threshold
        promising = trend >= slope_threshold

        block_state = np.full(mask.sum(), "Promiscuous", dtype=object)
        block_state[loyal[mask]] = "Loyal"
        block_state[promising[mask]] = "Promising"
        block_state[risky[mask]] = "Risky"
        state[mask] = block_state
    return pd.Series(state, index=spend_4w.index)


def add_future_projection(
    features: pd.DataFrame,
    weekly_sales: pd.DataFrame,
    product_forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    product_summary = (
        product_forecasts.groupby(["product_id", "category_h", "family_h", "analytical_block"], as_index=False)
        .agg(
            product_forecast_4w=("forecast_revenue", "sum"),
            product_recent_4w=("recent_4w_revenue", "first"),
            product_previous_4w=("previous_4w_revenue", "first"),
            product_forecast_factor=("product_forecast_factor", "first"),
            product_forecast_change_pct=("forecast_change_pct", "first"),
            historical_product_slope_pct=("historical_product_slope_pct", "first"),
            forecast_product_slope_pct=("forecast_product_slope_pct", "first"),
        )
    )
    category_factor = (
        product_summary.groupby(["category_h", "analytical_block"], as_index=False)
        .agg(category_forecast_4w=("product_forecast_4w", "sum"), category_recent_4w=("product_recent_4w", "sum"))
    )
    category_factor["category_forecast_factor"] = (
        category_factor["category_forecast_4w"] / category_factor["category_recent_4w"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.2, upper=2.5)

    last_weeks = sorted(weekly_sales["week_start"].dropna().unique())[-12:]
    last_4 = last_weeks[-4:]
    client_product = (
        weekly_sales[weekly_sales["week_start"].isin(last_weeks)]
        .groupby(["client_id", "category_h", "analytical_block", "product_id", "family_h"], as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "client_product_spend_12w"})
    )
    client_product_4 = (
        weekly_sales[weekly_sales["week_start"].isin(last_4)]
        .groupby(["client_id", "category_h", "analytical_block", "product_id", "family_h"], as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "client_product_spend_4w"})
    )
    client_product = client_product.merge(
        client_product_4,
        on=["client_id", "category_h", "analytical_block", "product_id", "family_h"],
        how="left",
    )
    client_product["client_product_spend_4w"] = client_product["client_product_spend_4w"].fillna(0.0)
    client_product = client_product.merge(
        product_summary[
            [
                "product_id",
                "product_forecast_factor",
                "product_forecast_change_pct",
                "historical_product_slope_pct",
                "forecast_product_slope_pct",
            ]
        ],
        on="product_id",
        how="left",
    )
    client_product["product_forecast_factor"] = client_product["product_forecast_factor"].fillna(1.0)
    client_product["product_forecast_change_pct"] = client_product["product_forecast_change_pct"].fillna(0.0)
    client_product["historical_product_slope_pct"] = client_product["historical_product_slope_pct"].fillna(0.0)
    client_product["forecast_product_slope_pct"] = client_product["forecast_product_slope_pct"].fillna(0.0)

    weighted = client_product.copy()
    weighted["weighted_factor"] = weighted["client_product_spend_12w"].clip(lower=0) * weighted["product_forecast_factor"]
    factor = (
        weighted.groupby(["client_id", "category_h", "analytical_block"], as_index=False)
        .agg(weighted_factor=("weighted_factor", "sum"), positive_spend_12w=("client_product_spend_12w", lambda s: s.clip(lower=0).sum()))
    )
    factor["client_product_forecast_factor"] = (
        factor["weighted_factor"] / factor["positive_spend_12w"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0.2, upper=2.5)

    slope_weighted = client_product.copy()
    slope_weighted["product_importance_weight"] = slope_weighted["client_product_spend_12w"].clip(lower=0)
    slope_weighted["weighted_historical_slope"] = slope_weighted["product_importance_weight"] * slope_weighted["historical_product_slope_pct"]
    slope_weighted["weighted_forecast_slope"] = slope_weighted["product_importance_weight"] * slope_weighted["forecast_product_slope_pct"]
    slope_factor = (
        slope_weighted.groupby(["client_id", "category_h", "analytical_block"], as_index=False)
        .agg(
            weighted_historical_slope=("weighted_historical_slope", "sum"),
            weighted_forecast_slope=("weighted_forecast_slope", "sum"),
            slope_weight_sum=("product_importance_weight", "sum"),
        )
    )
    slope_factor["client_weighted_historical_product_slope"] = (
        slope_factor["weighted_historical_slope"] / slope_factor["slope_weight_sum"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    slope_factor["client_weighted_forecast_product_slope"] = (
        slope_factor["weighted_forecast_slope"] / slope_factor["slope_weight_sum"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = features.merge(factor[["client_id", "category_h", "analytical_block", "client_product_forecast_factor"]], on=["client_id", "category_h", "analytical_block"], how="left")
    out = out.merge(
        slope_factor[
            [
                "client_id",
                "category_h",
                "analytical_block",
                "client_weighted_historical_product_slope",
                "client_weighted_forecast_product_slope",
            ]
        ],
        on=["client_id", "category_h", "analytical_block"],
        how="left",
    )
    out = out.merge(category_factor[["category_h", "analytical_block", "category_forecast_factor"]], on=["category_h", "analytical_block"], how="left")
    out["client_product_forecast_factor"] = out["client_product_forecast_factor"].fillna(out["category_forecast_factor"]).fillna(1.0)
    out["client_weighted_historical_product_slope"] = out["client_weighted_historical_product_slope"].fillna(0.0)
    out["client_weighted_forecast_product_slope"] = out["client_weighted_forecast_product_slope"].fillna(0.0)
    trend_factor = (1.0 + out["trend_4w_vs_prev_4w"].clip(lower=-0.50, upper=0.50)).fillna(1.0)
    base_spend = out["spend_4w"].copy()
    inactive_with_recent_history = (base_spend <= 0) & (out["prev_spend_4w"] > 0)
    inactive_with_annual_history = (base_spend <= 0) & (out["prev_spend_4w"] <= 0) & (out["spend_52w"] > 0)
    base_spend = base_spend.mask(inactive_with_recent_history, out["prev_spend_4w"] * 0.65)
    base_spend = base_spend.mask(inactive_with_annual_history, (out["spend_52w"] / 13.0) * 0.35)
    out["predicted_spend_4w"] = (base_spend * trend_factor * out["client_product_forecast_factor"]).clip(lower=0.0)
    out["predicted_spend_change_pct"] = [safe_pct_change(n, o) for n, o in zip(out["predicted_spend_4w"], out["spend_4w"])]

    driver = client_product.copy()
    driver["predicted_client_product_spend_4w"] = (
        driver["client_product_spend_4w"].clip(lower=0) * driver["product_forecast_factor"]
    )
    driver["driver_delta"] = driver["predicted_client_product_spend_4w"] - driver["client_product_spend_4w"]
    driver["driver_abs_delta"] = driver["driver_delta"].abs()
    driver = driver.sort_values("driver_abs_delta", ascending=False).drop_duplicates(["client_id", "category_h", "analytical_block"])
    driver = driver.rename(
        columns={
            "product_id": "driver_product_id",
            "family_h": "driver_family_h",
            "product_forecast_change_pct": "driver_product_forecast_change_pct",
            "historical_product_slope_pct": "driver_historical_product_slope_pct",
            "forecast_product_slope_pct": "driver_forecast_product_slope_pct",
        }
    )
    return out, driver


def build_alerts(classifications: pd.DataFrame, product_driver: pd.DataFrame) -> pd.DataFrame:
    alerts = classifications[classifications["classification_eligible"]].copy()
    alerts = alerts.merge(
        product_driver[
            [
                "client_id",
                "category_h",
                "analytical_block",
                "driver_product_id",
                "driver_family_h",
                "driver_delta",
                "driver_product_forecast_change_pct",
                "driver_historical_product_slope_pct",
                "driver_forecast_product_slope_pct",
            ]
        ],
        on=["client_id", "category_h", "analytical_block"],
        how="left",
    )
    alerts["potential_gap_4w"] = (alerts["potential_4w"] - alerts["predicted_spend_4w"]).clip(lower=0.0)
    alerts["state_changed"] = alerts["current_state"] != alerts["future_state"]
    alerts = alerts[alerts["state_changed"] | alerts["future_state"].eq("Risky")].copy()
    if alerts.empty:
        return alerts
    alerts["severity_weight"] = alerts.apply(alert_severity, axis=1)
    alerts["priority_score"] = (
        alerts["severity_weight"]
        * np.log1p(alerts["potential_gap_4w"] + (alerts["spend_4w"] - alerts["predicted_spend_4w"]).abs())
        * (0.5 + alerts["regularity_score"])
        * (1.0 + alerts["predicted_spend_change_pct"].abs().clip(upper=2.0))
    )
    alerts = alerts[(alerts["severity_weight"] > 0) & (alerts["priority_score"] > 0)].copy()
    if alerts.empty:
        return alerts
    high_cut = alerts["priority_score"].quantile(0.80)
    med_cut = alerts["priority_score"].quantile(0.50)
    alerts["priority"] = np.select(
        [alerts["priority_score"] >= high_cut, alerts["priority_score"] >= med_cut],
        ["High", "Medium"],
        default="Low",
    )
    alerts["explanation"] = alerts.apply(alert_explanation, axis=1)
    return alerts.sort_values(["priority_score", "potential_gap_4w"], ascending=False)


def build_transition_tables(classifications: pd.DataFrame) -> dict[str, pd.DataFrame]:
    eligible = classifications[classifications["classification_eligible"]].copy()
    outputs: dict[str, pd.DataFrame] = {}
    counts = pd.crosstab(eligible["current_state"], eligible["future_state"]).reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    outputs["transition_counts"] = counts
    outputs["transition_probabilities"] = probs
    for block, block_df in eligible.groupby("analytical_block"):
        block_counts = pd.crosstab(block_df["current_state"], block_df["future_state"]).reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
        block_probs = block_counts.div(block_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        safe_block = str(block).replace(" ", "_").replace("é", "e")
        outputs[f"transition_counts_{safe_block}"] = block_counts
        outputs[f"transition_probabilities_{safe_block}"] = block_probs
    return outputs


def alert_severity(row: pd.Series) -> float:
    current = row["current_state"]
    future = row["future_state"]
    if current == future and future != "Risky":
        return 0.0
    if future == "Risky" and current != "Risky":
        return 3.0
    if current == "Loyal" and future == "Promiscuous":
        return 2.5
    if future == "Promiscuous" and current in {"Loyal", "Promising"}:
        return 2.0
    if future == "Risky":
        return 1.7
    if future == "Promising" and current != "Promising":
        return 1.1
    if future == "Promiscuous" and row.get("potential_gap_4w", 0) > 0:
        return 0.8
    return 0.0


def alert_explanation(row: pd.Series) -> str:
    pct = row["predicted_spend_change_pct"] * 100.0
    capture = row["predicted_capture_ratio"] * 100.0
    slope = row.get("classification_forecast_slope", 0.0) * 100.0
    driver = row.get("driver_product_id")
    driver_text = "no single product driver"
    if pd.notna(driver):
        driver_text = (
            f"product {driver} "
            f"(forecast slope {row.get('driver_forecast_product_slope_pct', 0) * 100:.1f}%, "
            f"forecast change {row.get('driver_product_forecast_change_pct', 0) * 100:.1f}%)"
        )
    return (
        f"{row['current_state']} -> {row['future_state']} in {row['category_h']}. "
        f"Projected 4-week spend changes {pct:.1f}%, product-weighted slope is {slope:.1f}%, "
        f"and spend reaches {capture:.1f}% of potential. "
        f"Main driver: {driver_text}."
    )


def write_outputs(
    output_dir: Path,
    weekly_sales: pd.DataFrame,
    type_forecasts: pd.DataFrame,
    product_forecasts: pd.DataFrame,
    classifications: pd.DataFrame,
    alerts: pd.DataFrame,
    summary: dict[str, Any],
    top_alerts: int,
    transitions: dict[str, pd.DataFrame] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    weekly_sales.to_csv(output_dir / "weekly_sales.csv", index=False)
    type_forecasts.to_csv(output_dir / "type_forecasts.csv", index=False)
    product_forecasts.to_csv(output_dir / "product_forecasts.csv", index=False)
    classifications.to_csv(output_dir / "client_classifications.csv", index=False)
    alerts.to_csv(output_dir / "alerts.csv", index=False)
    alerts.head(top_alerts).to_csv(output_dir / "top_alerts.csv", index=False)
    if transitions:
        for name, table in transitions.items():
            table.to_csv(output_dir / f"{name}.csv")
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    try:
        with pd.ExcelWriter(output_dir / "smart_demand_outputs.xlsx") as writer:
            alerts.head(top_alerts).to_excel(writer, sheet_name="Top Alerts", index=False)
            classifications.head(5000).to_excel(writer, sheet_name="Classifications", index=False)
            type_forecasts.to_excel(writer, sheet_name="Type Forecasts", index=False)
            product_forecasts.to_excel(writer, sheet_name="Product Forecasts", index=False)
            if transitions:
                for name, table in transitions.items():
                    sheet_name = name[:31]
                    table.to_excel(writer, sheet_name=sheet_name)
            pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
    except Exception as exc:  # pragma: no cover - optional Excel writer path
        (output_dir / "excel_export_error.txt").write_text(str(exc), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")
    classification_config = load_classification_config(args.classification_config_json)
    model_config = load_model_config(args.model_config_json)

    sheets = load_workbook(input_path)
    data = prepare_data(sheets)
    current_week, history_weeks, future_weeks, all_weeks = build_week_calendar(
        data["sales"], args.current_end, args.forecast_weeks
    )
    history_weeks = apply_history_limit(history_weeks, args.history_weeks)
    all_weeks = history_weeks.union(future_weeks)
    campaigns = build_campaign_calendar(data["campaigns"], all_weeks)
    weekly_sales = build_weekly_sales(data["sales"], campaigns)
    weekly_sales = weekly_sales[weekly_sales["week_start"].isin(history_weeks)]

    type_forecasts, factor_by_block = forecast_type_sales(
        weekly_sales, history_weeks, future_weeks, args.model_mode, args.maxiter, model_config
    )
    product_forecasts = forecast_product_sales(
        weekly_sales,
        data["products"],
        history_weeks,
        future_weeks,
        factor_by_block,
        args.model_mode,
        args.maxiter,
        model_config,
    )
    classifications, product_driver = build_client_features(
        weekly_sales, data["potential"], current_week, history_weeks, product_forecasts, classification_config
    )
    alerts = build_alerts(classifications, product_driver)
    transitions = build_transition_tables(classifications)

    summary = {
        "input": str(input_path),
        "current_week": current_week,
        "forecast_weeks": args.forecast_weeks,
        "history_weeks": len(history_weeks),
        "sales_rows": len(data["sales"]),
        "weekly_sales_rows": len(weekly_sales),
        "clients_with_sales": int(data["sales"]["client_id"].nunique()),
        "products": int(data["products"]["product_id"].nunique()),
        "classification_rows": len(classifications),
        "classification_eligible_rows": int(classifications["classification_eligible"].sum()),
        "relationship_status_counts": classifications["relationship_status"].value_counts().to_dict(),
        "current_state_counts": classifications["current_state"].value_counts().to_dict(),
        "future_state_counts": classifications["future_state"].value_counts().to_dict(),
        "alerts": len(alerts),
        "high_priority_alerts": int((alerts.get("priority") == "High").sum()) if not alerts.empty else 0,
        "statsmodels_available": HAS_STATSMODELS,
        "model_mode": args.model_mode,
        "classification_config": classification_config,
        "model_config": model_config,
    }
    write_outputs(
        output_dir,
        weekly_sales,
        type_forecasts,
        product_forecasts,
        classifications,
        alerts,
        summary,
        args.top_alerts,
        transitions,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
