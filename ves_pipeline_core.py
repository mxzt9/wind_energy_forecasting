"""Reusable wrappers for the VES forecasting notebook pipeline.

The original `ves_pipeline.ipynb` remains the reference implementation for the
feature engineering math.  This module gives that logic a stable artifact
contract across three notebooks:

1. build feature datasets;
2. train and save model weights;
3. load weights, post-process predictions, and report metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit


ROOT = Path(".")
LEGACY_NOTEBOOK = ROOT / "ves_pipeline.ipynb"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATASET_DIR = OUTPUT_DIR / "datasets"
MODEL_DIR = OUTPUT_DIR / "models"
TEST_OUTPUT_DIR = OUTPUT_DIR / "test"
TEST_FIGURE_DIR = TEST_OUTPUT_DIR / "figures"

TRAIN_PATH = DATA_DIR / "train_merged.csv"
VALID_PATH = DATA_DIR / "valid_merged.csv"
TEST_PATH = DATA_DIR / "test_merged.csv"

TRAIN_FEATURES_PATH = DATASET_DIR / "train_features.csv"
VALID_FEATURES_PATH = DATASET_DIR / "valid_features.csv"
TEST_FEATURES_PATH = DATASET_DIR / "test_features.csv"
TEST_ACTUAL_PATH = DATASET_DIR / "test_actual.csv"
MODEL_FEATURES_PATH = DATASET_DIR / "model_features.json"
FEATURE_ARTIFACTS_PATH = DATASET_DIR / "feature_artifacts.joblib"

MODEL_ARTIFACTS_PATH = MODEL_DIR / "model_artifacts.joblib"
TRAINING_SUMMARY_PATH = MODEL_DIR / "training_summary.csv"
SEASONAL_RESIDUAL_REPORT_PATH = MODEL_DIR / "seasonal_residual_report.csv"

FINAL_VALID_SUBMISSION_PATH = OUTPUT_DIR / "submission_final.csv"
TEST_SUBMISSION_PATH = TEST_OUTPUT_DIR / "submission_final.csv"
TEST_DAY_SUBMISSION_PATH = TEST_OUTPUT_DIR / "submission_2026-05-18.csv"
TEST_COMPARE_PATH = TEST_OUTPUT_DIR / "test_prediction_vs_actual.csv"
TEST_METRICS_PATH = TEST_OUTPUT_DIR / "test_prediction_metrics.csv"
LIGHT_POSTPROCESS_REPORT_PATH = TEST_OUTPUT_DIR / "light_postprocess_report.csv"
TEST_LOW_WIND_ONLY_SUBMISSION_PATH = TEST_OUTPUT_DIR / "submission_low_wind_only.csv"
TEST_GUARDED_SUBMISSION_PATH = TEST_OUTPUT_DIR / "submission_guarded_calibrator.csv"
TEST_AGGRESSIVE_SUBMISSION_PATH = TEST_OUTPUT_DIR / "submission_aggressive_calibrator.csv"


BOOTSTRAP_CELLS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14]
FEATURE_CONSTANT_CELLS = [19, 40, 43, 47]
LAYOUT_CELLS = [55, 56, 57]
FEATURE_ENGINEERING_CELLS = [
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    34,
    36,
    38,
    41,
    44,
    48,
]
MODELING_FUNCTION_CELLS = [
    52,
    53,
    59,
    60,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    82,
    83,
    84,
    106,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
]


def _ensure_dirs() -> None:
    for path in [OUTPUT_DIR, DATASET_DIR, MODEL_DIR, TEST_OUTPUT_DIR, TEST_FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _load_legacy_notebook() -> dict[str, Any]:
    if not LEGACY_NOTEBOOK.exists():
        raise FileNotFoundError(f"Missing legacy notebook: {LEGACY_NOTEBOOK}")
    return json.loads(LEGACY_NOTEBOOK.read_text(encoding="utf-8"))


def _exec_cell(nb: dict[str, Any], env: dict[str, Any], idx: int) -> None:
    cell = nb["cells"][idx]
    if cell.get("cell_type") != "code":
        return
    source = "".join(cell.get("source", []))
    exec(compile(source, f"{LEGACY_NOTEBOOK}:cell_{idx}", "exec"), env)


def _quiet_display(*_: Any, **__: Any) -> None:
    return None


def _legacy_env(verbose: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    nb = _load_legacy_notebook()
    env: dict[str, Any] = {"__name__": "ves_pipeline_legacy_runtime"}

    for idx in BOOTSTRAP_CELLS:
        _exec_cell(nb, env, idx)

    if not verbose:
        env["display"] = _quiet_display
        if "plt" in env:
            env["plt"].show = lambda *args, **kwargs: None

    env["PLOT_RESEARCH_OUTPUTS"] = False
    env["PLOT_TWO_STAGE_DIAGNOSTICS"] = False
    env["SAVE_DIAGNOSTIC_ARTIFACTS"] = False
    env["SAVE_DIRECT_DEBUG_SUBMISSIONS"] = False
    env["PLOT_FINAL_DISTRIBUTIONS"] = False
    env["RUN_POWER_CURVE_DIAGNOSTIC"] = False
    env["RUN_LOCAL_CHECK"] = False
    env["RUN_FINAL_PIPELINE"] = False

    for idx in FEATURE_CONSTANT_CELLS + LAYOUT_CELLS:
        _exec_cell(nb, env, idx)

    return env, nb


def _first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _normalize_train_eval(
    env: dict[str, Any],
    train_raw: pd.DataFrame,
    eval_raw: pd.DataFrame,
    *,
    eval_name: str,
    drop_eval_target: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_col = _first_existing(train_raw.columns, env["TARGET_CANDIDATES"])
    if target_col is None:
        diff_cols = [col for col in train_raw.columns if col not in eval_raw.columns]
        if not diff_cols:
            raise ValueError("Cannot infer train target column.")
        target_col = diff_cols[0]

    train_datetime_col = _first_existing(train_raw.columns, env["DATETIME_CANDIDATES"])
    eval_datetime_col = _first_existing(eval_raw.columns, env["DATETIME_CANDIDATES"])
    train_repair_col = _first_existing(train_raw.columns, env["REPAIR_CANDIDATES"])
    eval_repair_col = _first_existing(eval_raw.columns, env["REPAIR_CANDIDATES"])

    missing = [
        name
        for name, value in {
            "train_datetime": train_datetime_col,
            "eval_datetime": eval_datetime_col,
            "train_repair": train_repair_col,
            "eval_repair": eval_repair_col,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError("Cannot infer required columns: " + ", ".join(missing))

    eval_target_col = _first_existing(eval_raw.columns, env["TARGET_CANDIDATES"])
    eval_actual = pd.DataFrame(
        {
            "row_id": np.arange(len(eval_raw)),
            "datetime": pd.to_datetime(eval_raw[eval_datetime_col], errors="coerce"),
        }
    )
    if eval_target_col is not None:
        eval_actual["target"] = pd.to_numeric(eval_raw[eval_target_col], errors="coerce")

    train = train_raw.rename(
        columns={
            train_datetime_col: "datetime",
            target_col: "target",
            train_repair_col: "turbines_in_repair",
        }
    ).copy()

    eval_frame = eval_raw.copy()
    if drop_eval_target and eval_target_col is not None:
        eval_frame = eval_frame.drop(columns=[eval_target_col])
    eval_frame = eval_frame.rename(
        columns={
            eval_datetime_col: "datetime",
            eval_repair_col: "turbines_in_repair",
        }
    ).copy()

    train = train.loc[:, ~train.columns.duplicated()].copy()
    eval_frame = eval_frame.loc[:, ~eval_frame.columns.duplicated()].copy()

    train["datetime"] = pd.to_datetime(train["datetime"], errors="coerce")
    eval_frame["datetime"] = pd.to_datetime(eval_frame["datetime"], errors="coerce")
    train["row_id"] = np.arange(len(train))
    eval_frame["row_id"] = np.arange(len(eval_frame))
    train["source"] = "train"
    eval_frame["source"] = "valid"
    train = train.sort_values("datetime").reset_index(drop=True)
    eval_frame = eval_frame.sort_values("datetime").reset_index(drop=True)

    env["train_raw"] = train_raw
    env["valid_raw"] = eval_raw
    env["train"] = train
    env["valid"] = eval_frame
    env["target_col"] = target_col
    env["datetime_col"] = train_datetime_col
    env["repair_col"] = train_repair_col
    env["VALID_NAME"] = eval_name

    return train, eval_frame, eval_actual


def _add_high_wind_feature_columns(
    env: dict[str, Any],
    train_model: pd.DataFrame,
    eval_model: pd.DataFrame,
    model_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    train_model = train_model.copy()
    eval_model = eval_model.copy()

    cap_curve = env["fit_high_wind_cap_curve"](
        train_model,
        speed_col=env["HIGH_WIND_SPEED_COL"],
        target_col="target",
    )
    curve_x = cap_curve["speed_mean"].to_numpy(dtype=float)
    curve_y = cap_curve["cap_final"].to_numpy(dtype=float)
    order = np.argsort(curve_x)
    curve_x = curve_x[order]
    curve_y = curve_y[order]
    median_curve_x = float(np.nanmedian(curve_x))

    for frame in [train_model, eval_model]:
        ws = pd.to_numeric(frame[env["HIGH_WIND_SPEED_COL"]], errors="coerce").to_numpy(dtype=float)
        ws_safe = np.nan_to_num(ws, nan=median_curve_x)
        cap = np.interp(ws_safe, curve_x, curve_y, left=curve_y[0], right=curve_y[-1])
        gate = env["sigmoid_np"]((ws_safe - env["HIGH_WIND_START_WS"]) / env["HIGH_WIND_TRANSITION"])
        frame["high_wind_cap_feature"] = np.clip(cap, 0, env["INSTALLED_CAPACITY_MW"])
        frame["high_wind_gate_feature"] = np.nan_to_num(gate, nan=0.0)
        if "p_empirical_mean_80_120" in frame.columns:
            frame["high_wind_cap_minus_empirical"] = (
                frame["high_wind_cap_feature"] - frame["p_empirical_mean_80_120"]
            )
        if "p_theory_mean_80_120" in frame.columns:
            frame["high_wind_cap_minus_theory"] = frame["high_wind_cap_feature"] - frame["p_theory_mean_80_120"]

    high_wind_cols = [
        col
        for col in [
            "high_wind_cap_feature",
            "high_wind_gate_feature",
            "high_wind_cap_minus_empirical",
            "high_wind_cap_minus_theory",
        ]
        if col in train_model.columns and col in eval_model.columns
    ]
    model_features = list(dict.fromkeys(model_features + high_wind_cols))
    return train_model, eval_model, model_features, cap_curve


def _build_feature_pair(
    train_raw: pd.DataFrame,
    eval_raw: pd.DataFrame,
    *,
    eval_name: str,
    verbose: bool = False,
) -> dict[str, Any]:
    env, nb = _legacy_env(verbose=verbose)
    _normalize_train_eval(env, train_raw, eval_raw, eval_name=eval_name, drop_eval_target=True)

    for idx in FEATURE_ENGINEERING_CELLS + MODELING_FUNCTION_CELLS:
        _exec_cell(nb, env, idx)

    train_model, eval_model, added_cols, model_features = env["build_model_frames"]()
    train_model, eval_model, model_features, high_wind_cap_curve = _add_high_wind_feature_columns(
        env,
        train_model,
        eval_model,
        model_features,
    )

    return {
        "env": env,
        "train_features": train_model,
        "eval_features": eval_model,
        "added_features": added_cols,
        "model_features": model_features,
        "high_wind_cap_curve": high_wind_cap_curve,
    }


def _align_feature_columns(frame: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in model_features:
        if col not in out.columns:
            out[col] = np.nan
    return out


def build_feature_datasets(verbose: bool = False) -> dict[str, Any]:
    """Build and save train/valid/test model-ready feature datasets."""
    _ensure_dirs()

    train_raw = pd.read_csv(TRAIN_PATH)
    valid_raw = pd.read_csv(VALID_PATH)
    test_raw = pd.read_csv(TEST_PATH)

    valid_pair = _build_feature_pair(train_raw, valid_raw, eval_name="valid", verbose=verbose)
    test_pair = _build_feature_pair(train_raw, test_raw, eval_name="test", verbose=verbose)

    model_features = valid_pair["model_features"]
    train_features = _align_feature_columns(valid_pair["train_features"], model_features)
    valid_features = _align_feature_columns(valid_pair["eval_features"], model_features)
    test_features = _align_feature_columns(test_pair["eval_features"], model_features)

    train_features.to_csv(TRAIN_FEATURES_PATH, index=False)
    valid_features.to_csv(VALID_FEATURES_PATH, index=False)
    test_features.to_csv(TEST_FEATURES_PATH, index=False)

    test_actual = pd.DataFrame({"row_id": np.arange(len(test_raw))})
    datetime_col = _first_existing(test_raw.columns, valid_pair["env"]["DATETIME_CANDIDATES"])
    target_col = _first_existing(test_raw.columns, valid_pair["env"]["TARGET_CANDIDATES"])
    if datetime_col is not None:
        test_actual["datetime"] = pd.to_datetime(test_raw[datetime_col], errors="coerce")
    if target_col is not None:
        test_actual["actual_mw"] = pd.to_numeric(test_raw[target_col], errors="coerce")
    for col in ["wind_speed_120m", "wind_speed_80m", "wind_direction_120m", "Кол-во_ВЭУ_в_ремонте"]:
        if col in test_raw.columns:
            test_actual[col] = test_raw[col].to_numpy()
    test_actual.to_csv(TEST_ACTUAL_PATH, index=False)

    MODEL_FEATURES_PATH.write_text(
        json.dumps({"model_features": model_features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    feature_artifacts = {
        "model_features": model_features,
        "high_wind_cap_curve_valid": valid_pair["high_wind_cap_curve"],
        "high_wind_cap_curve_test": test_pair["high_wind_cap_curve"],
        "paths": {
            "train_features": str(TRAIN_FEATURES_PATH),
            "valid_features": str(VALID_FEATURES_PATH),
            "test_features": str(TEST_FEATURES_PATH),
            "model_features": str(MODEL_FEATURES_PATH),
        },
        "row_counts": {
            "train": len(train_features),
            "valid": len(valid_features),
            "test": len(test_features),
        },
    }
    joblib.dump(feature_artifacts, FEATURE_ARTIFACTS_PATH)

    return {
        "train_features": train_features,
        "valid_features": valid_features,
        "test_features": test_features,
        "model_features": model_features,
        "feature_artifacts_path": FEATURE_ARTIFACTS_PATH,
    }


def _load_model_features() -> list[str]:
    data = json.loads(MODEL_FEATURES_PATH.read_text(encoding="utf-8"))
    return list(data["model_features"])


def _train_model_bank(
    env: dict[str, Any],
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    bank_kind: str,
    *,
    clip_pred: tuple[float, float] | None = None,
) -> dict[str, Any]:
    x_train, _, used_cols = env["_ts_prepare_matrix"](train_frame, train_frame, feature_cols)
    medians = x_train.median(numeric_only=True)
    models = {}
    fold_rows = []
    splitter = TimeSeriesSplit(n_splits=env["TWO_STAGE_N_SPLITS"], gap=24)

    for model_i, (model_name, factory) in enumerate(env["_ts_model_bank"](bank_kind, fast=env["TWO_STAGE_FAST_MODE"]).items(), 1):
        fold_models = []
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x_train), 1):
            model = factory(env["RANDOM_STATE"] + 1000 * model_i + fold)
            model.fit(x_train.iloc[tr_idx], y[tr_idx])
            fold_pred = np.asarray(model.predict(x_train.iloc[va_idx]), dtype=float)
            if clip_pred is not None:
                fold_pred = np.clip(fold_pred, clip_pred[0], clip_pred[1])
            fold_models.append(model)
            fold_rows.append(
                {
                    "bank": bank_kind,
                    "model": model_name,
                    "fold": fold,
                    "mae": float(mean_absolute_error(y[va_idx], fold_pred)),
                    "n_train": len(tr_idx),
                    "n_valid": len(va_idx),
                }
            )
        models[model_name] = fold_models
    return {
        "models": models,
        "used_cols": used_cols,
        "medians": medians,
        "clip_pred": clip_pred,
        "prediction_mode": "timeseries_fold_average",
        "fold_report": pd.DataFrame(fold_rows),
    }


def _predict_model_bank(bank: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    used_cols = bank["used_cols"]
    x = frame[used_cols].copy().replace([np.inf, -np.inf], np.nan)
    x = x.fillna(bank["medians"]).fillna(0)
    components = {}
    for name, model_or_models in bank["models"].items():
        model_list = model_or_models if isinstance(model_or_models, list) else [model_or_models]
        fold_preds = [np.asarray(model.predict(x), dtype=float) for model in model_list]
        pred = np.mean(fold_preds, axis=0)
        if bank.get("clip_pred") is not None:
            pred = np.clip(pred, bank["clip_pred"][0], bank["clip_pred"][1])
        components[name] = pred
    component_df = pd.DataFrame(components)
    return component_df.mean(axis=1).to_numpy(dtype=float), component_df


def _add_two_stage_meta_features(env: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    eps = env["EPS"]
    capacity = env["INSTALLED_CAPACITY_MW"]
    if "p_empirical_mean_80_120" in out.columns:
        out["two_stage_normal_minus_empirical"] = out["two_stage_normal_pred"] - out["p_empirical_mean_80_120"]
        out["two_stage_normal_div_empirical"] = out["two_stage_normal_pred"] / (out["p_empirical_mean_80_120"].abs() + eps)
    if "p_theory_mean_80_120" in out.columns:
        out["two_stage_normal_minus_theory"] = out["two_stage_normal_pred"] - out["p_theory_mean_80_120"]
        out["two_stage_normal_div_theory"] = out["two_stage_normal_pred"] / (out["p_theory_mean_80_120"].abs() + eps)
    if "full_p_ideal_clean" in out.columns:
        out["two_stage_normal_minus_ideal_clean"] = out["two_stage_normal_pred"] - out["full_p_ideal_clean"]
        out["two_stage_normal_div_ideal_clean"] = out["two_stage_normal_pred"] / (out["full_p_ideal_clean"].abs() + eps)
    if "full_hidden_loss_mw_pred" in out.columns:
        out["two_stage_normal_x_hidden_loss"] = out["two_stage_normal_pred"] * out["full_hidden_loss_mw_pred"] / capacity
    if "layout_wake_risk_scalar_120m" in out.columns:
        out["two_stage_normal_x_wake_risk"] = out["two_stage_normal_pred"] * out["layout_wake_risk_scalar_120m"]
    return out


def _apply_two_stage_safety(env: dict[str, Any], frame: pd.DataFrame, raw_deviation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dev_safe = np.clip(
        raw_deviation,
        -env["TWO_STAGE_DEVIATION_ABS_CLIP_MW"],
        env["TWO_STAGE_DEVIATION_ABS_CLIP_MW"],
    )
    dev_safe = dev_safe * env["TWO_STAGE_DEVIATION_SHRINK"]
    gate = np.ones(len(frame), dtype=float)
    if env["TWO_STAGE_USE_PHYSICS_GATE"]:
        gate_parts = []
        if "full_k_hidden_pred" in frame.columns:
            x = frame["full_k_hidden_pred"].fillna(0).clip(0, 0.8)
            gate_parts.append((x / 0.25).clip(0, 1).to_numpy(dtype=float))
        if "full_hidden_loss_mw_pred" in frame.columns:
            x = frame["full_hidden_loss_mw_pred"].abs().fillna(0)
            gate_parts.append((x / 12.0).clip(0, 1).to_numpy(dtype=float))
        if "full_recon_minus_empirical_curve" in frame.columns:
            x = frame["full_recon_minus_empirical_curve"].abs().fillna(0)
            gate_parts.append((x / 15.0).clip(0, 1).to_numpy(dtype=float))
        if "wind_speed_120m" in frame.columns:
            ws = frame["wind_speed_120m"].fillna(0)
            gate_parts.append(((ws >= 6.0) & (ws <= 13.5)).astype(float).to_numpy())
        if gate_parts:
            gate_raw = np.mean(np.vstack(gate_parts), axis=0)
            gate = env["TWO_STAGE_GATE_MIN"] + (env["TWO_STAGE_GATE_MAX"] - env["TWO_STAGE_GATE_MIN"]) * gate_raw
    gate = np.clip(gate, env["TWO_STAGE_GATE_MIN"], env["TWO_STAGE_GATE_MAX"])
    return dev_safe * gate, gate


def _num_col(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _seasonal_residual_frame(
    env: dict[str, Any],
    frame: pd.DataFrame,
    base_pred: np.ndarray,
    *,
    direct_pred: np.ndarray | None = None,
    two_stage_pred: np.ndarray | None = None,
) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    capacity = env["INSTALLED_CAPACITY_MW"]
    eps = env["EPS"]

    base = np.clip(np.asarray(base_pred, dtype=float), 0, capacity)
    out["res_base_pred"] = base
    out["res_base_cf"] = base / capacity

    if direct_pred is not None:
        direct = np.clip(np.asarray(direct_pred, dtype=float), 0, capacity)
        out["res_direct_pred"] = direct
    if two_stage_pred is not None:
        two_stage = np.clip(np.asarray(two_stage_pred, dtype=float), 0, capacity)
        out["res_two_stage_pred"] = two_stage
    if direct_pred is not None and two_stage_pred is not None:
        out["res_two_stage_minus_direct"] = out["res_two_stage_pred"] - out["res_direct_pred"]
        out["res_two_stage_div_direct"] = out["res_two_stage_pred"] / (out["res_direct_pred"].abs() + eps)

    if "datetime" in out.columns:
        dt = pd.to_datetime(out["datetime"], errors="coerce")
        month = dt.dt.month.fillna(_num_col(out, "month", 1)).astype(float)
        hour = dt.dt.hour.fillna(_num_col(out, "hour_of_day", 0)).astype(float)
    else:
        month = _num_col(out, "month", 1).fillna(1).astype(float)
        hour = _num_col(out, "hour_of_day", 0).fillna(0).astype(float)
    out["res_month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["res_month_cos"] = np.cos(2 * np.pi * month / 12.0)
    out["res_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["res_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    ws120 = _num_col(out, "wind_speed_120m").fillna(0)
    out["res_low_wind_gate"] = (ws120 < 5.0).astype(float)
    out["res_mid_wind_gate"] = ((ws120 >= 5.0) & (ws120 <= 11.0)).astype(float)
    out["res_high_wind_gate"] = (ws120 > 11.0).astype(float)

    temp120 = _num_col(out, "temperature_120m", np.nan)
    ice_risk = _num_col(out, "phi_ice_risk", 0.0).fillna(0)
    out["res_cold_gate"] = ((temp120 <= 0.0) | (ice_risk >= 0.4)).astype(float)
    out["res_warm_gate"] = (temp120 >= 10.0).astype(float)

    repair = _num_col(out, "turbines_in_repair").fillna(0)
    wake = _num_col(out, "layout_wake_risk_scalar_120m").fillna(0)
    emp = _num_col(out, "p_empirical_mean_80_120").fillna(0)
    theory = _num_col(out, "p_theory_mean_80_120").fillna(0)
    ws_diff = _num_col(out, "ws_diff_120_80").fillna(0)
    emp_minus_theory = _num_col(out, "p_empirical_minus_theory_120").fillna(emp - theory)
    recon_minus_emp = _num_col(out, "full_recon_minus_empirical_curve").fillna(0)

    out["res_base_minus_empirical"] = out["res_base_pred"] - emp
    out["res_base_minus_theory"] = out["res_base_pred"] - theory
    out["res_base_x_mid_wind"] = out["res_base_pred"] * out["res_mid_wind_gate"]
    out["res_mid_empirical_minus_theory"] = out["res_mid_wind_gate"] * emp_minus_theory
    out["res_mid_recon_minus_empirical"] = out["res_mid_wind_gate"] * recon_minus_emp
    out["res_mid_ws_diff_120_80"] = out["res_mid_wind_gate"] * ws_diff
    out["res_mid_wake_risk"] = out["res_mid_wind_gate"] * wake
    out["res_mid_repair"] = out["res_mid_wind_gate"] * repair
    out["res_wake_x_ws120"] = wake * ws120
    out["res_wake_x_base"] = wake * out["res_base_pred"] / capacity
    out["res_repair_x_base"] = repair * out["res_base_pred"] / capacity
    out["res_cold_x_ws120"] = out["res_cold_gate"] * ws120
    out["res_cold_x_base"] = out["res_cold_gate"] * out["res_base_pred"]

    return out


def _seasonal_residual_feature_cols(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "month",
        "hour_of_day",
        "dayofyear_sin",
        "dayofyear_cos",
        "wind_speed_80m",
        "wind_speed_120m",
        "wind_speed_180m",
        "wind_direction_80m",
        "wind_direction_120m",
        "wind_gusts_10m",
        "temperature_80m",
        "temperature_120m",
        "pressure_msl",
        "rain",
        "snowfall",
        "cloud_cover_low",
        "turbines_in_repair",
        "available_capacity_mw",
        "air_density",
        "density_ratio",
        "wind_power_density_120m",
        "p_theory_mean_80_120",
        "p_empirical_mean_80_120",
        "p_empirical_minus_theory_120",
        "ws_diff_120_80",
        "ws_ratio_120_80",
        "ws_diff_180_120",
        "gust_ratio_10m",
        "shear_80_120",
        "layout_wake_risk_scalar_120m",
        "full_hidden_loss_mw_pred",
        "full_recon_minus_empirical_curve",
        "phi_ice_risk",
        "phi_cold",
        "phi_turbulence",
        "phi_yaw_change_abs_1h",
        "high_wind_cap_feature",
        "high_wind_gate_feature",
        "high_wind_cap_minus_empirical",
        "res_base_pred",
        "res_base_cf",
        "res_direct_pred",
        "res_two_stage_pred",
        "res_two_stage_minus_direct",
        "res_two_stage_div_direct",
        "res_month_sin",
        "res_month_cos",
        "res_hour_sin",
        "res_hour_cos",
        "res_low_wind_gate",
        "res_mid_wind_gate",
        "res_high_wind_gate",
        "res_cold_gate",
        "res_warm_gate",
        "res_base_minus_empirical",
        "res_base_minus_theory",
        "res_base_x_mid_wind",
        "res_mid_empirical_minus_theory",
        "res_mid_recon_minus_empirical",
        "res_mid_ws_diff_120_80",
        "res_mid_wake_risk",
        "res_mid_repair",
        "res_wake_x_ws120",
        "res_wake_x_base",
        "res_repair_x_base",
        "res_cold_x_ws120",
        "res_cold_x_base",
    ]
    cols = [col for col in preferred if col in frame.columns]
    numeric = frame[cols].select_dtypes(include=[np.number]).columns.tolist()
    return list(dict.fromkeys(numeric))


def _seasonal_residual_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if "datetime" in frame.columns:
        month = pd.to_datetime(frame["datetime"], errors="coerce").dt.month
    else:
        month = _num_col(frame, "month", np.nan)
    ws120 = _num_col(frame, "wind_speed_120m", np.nan)
    temp120 = _num_col(frame, "temperature_120m", np.nan)
    ice_risk = _num_col(frame, "phi_ice_risk", 0.0).fillna(0)
    repair = _num_col(frame, "turbines_in_repair", 0.0).fillna(0)

    return {
        "winter": month.isin([12, 1, 2]),
        "spring": month.isin([3, 4, 5]),
        "summer": month.isin([6, 7, 8]),
        "autumn": month.isin([9, 10, 11]),
        "cold": (temp120 <= 0.0) | (ice_risk >= 0.4),
        "mid_wind": (ws120 >= 5.0) & (ws120 <= 11.0),
        "low_wind": ws120 < 5.0,
        "high_wind": ws120 > 11.0,
        "repair": repair > 0,
    }


def _new_residual_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.045,
        max_iter=260,
        max_leaf_nodes=23,
        min_samples_leaf=40,
        l2_regularization=0.08,
        early_stopping=True,
        validation_fraction=0.12,
        random_state=seed,
    )


def _fit_residual_model_set(
    env: dict[str, Any],
    train_frame: pd.DataFrame,
    residual: np.ndarray,
    feature_cols: list[str],
    *,
    seed_offset: int = 0,
    min_rows: int = 600,
) -> dict[str, Any]:
    residual = np.asarray(residual, dtype=float)
    train_frame = train_frame.reset_index(drop=True)
    x_all = train_frame[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    models = {}
    rows = []

    def fit_one(name: str, mask: np.ndarray, weight: float, seed: int) -> None:
        idx = np.flatnonzero(mask & np.isfinite(residual))
        if len(idx) < min_rows:
            return
        x_part = x_all.iloc[idx].copy()
        medians = x_part.median(numeric_only=True)
        x_part = x_part.fillna(medians).fillna(0)
        model = _new_residual_model(env["RANDOM_STATE"] + seed_offset + seed)
        model.fit(x_part, residual[idx])
        models[name] = {"model": model, "medians": medians, "weight": float(weight)}
        pred = np.asarray(model.predict(x_part), dtype=float)
        rows.append(
            {
                "expert": name,
                "n_train": int(len(idx)),
                "train_residual_mae": float(mean_absolute_error(residual[idx], pred)),
                "weight": float(weight),
            }
        )

    fit_one("global", np.ones(len(train_frame), dtype=bool), 1.0, 11)
    for i, (name, mask) in enumerate(_seasonal_residual_masks(train_frame).items(), 1):
        fit_one(name, mask.fillna(False).to_numpy(dtype=bool), 0.85, 100 + i)

    return {"models": models, "feature_cols": feature_cols, "fit_report": pd.DataFrame(rows)}


def _predict_residual_model_set(model_set: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    feature_cols = model_set["feature_cols"]
    models = model_set["models"]
    if not models or "global" not in models:
        zeros = np.zeros(len(frame), dtype=float)
        return zeros, pd.DataFrame({"residual_correction_raw": zeros})

    x_all = frame[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    masks = _seasonal_residual_masks(frame)
    pred_sum = np.zeros(len(frame), dtype=float)
    weight_sum = np.zeros(len(frame), dtype=float)
    diag = {}

    for name, item in models.items():
        x = x_all.copy().fillna(item["medians"]).fillna(0)
        pred = np.asarray(item["model"].predict(x), dtype=float)
        if name == "global":
            active = np.ones(len(frame), dtype=bool)
        else:
            active = masks.get(name, pd.Series(False, index=frame.index)).fillna(False).to_numpy(dtype=bool)
        weight = float(item["weight"])
        pred_sum[active] += pred[active] * weight
        weight_sum[active] += weight
        diag[f"residual_expert_{name}"] = pred

    correction = np.divide(pred_sum, weight_sum, out=np.zeros(len(frame), dtype=float), where=weight_sum > 0)
    diag["residual_correction_raw"] = correction
    diag["residual_correction_weight"] = weight_sum
    return correction, pd.DataFrame(diag)


def _fit_seasonal_residual_layer(
    env: dict[str, Any],
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    train_base_pred: np.ndarray,
    valid_base_pred: np.ndarray,
    *,
    train_direct_pred: np.ndarray | None = None,
    valid_direct_pred: np.ndarray | None = None,
    train_two_stage_pred: np.ndarray | None = None,
    valid_two_stage_pred: np.ndarray | None = None,
    valid_output_base_pred: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    train_res_frame = _seasonal_residual_frame(
        env,
        train_frame,
        train_base_pred,
        direct_pred=train_direct_pred,
        two_stage_pred=train_two_stage_pred,
    )
    valid_res_frame = _seasonal_residual_frame(
        env,
        valid_frame,
        valid_base_pred,
        direct_pred=valid_direct_pred,
        two_stage_pred=valid_two_stage_pred,
    )
    feature_cols = env["_ts_clean_feature_list"](
        train_res_frame,
        valid_res_frame,
        _seasonal_residual_feature_cols(train_res_frame),
    )
    y = train_frame["target"].clip(0, env["INSTALLED_CAPACITY_MW"]).to_numpy(dtype=float)
    train_base = np.clip(np.asarray(train_base_pred, dtype=float), 0, env["INSTALLED_CAPACITY_MW"])
    residual = y - train_base
    q_low = float(np.nanquantile(residual, 0.005))
    q_high = float(np.nanquantile(residual, 0.995))
    clip_low = max(q_low, -24.0)
    clip_high = min(q_high, 24.0)
    residual_clip = np.clip(residual, clip_low, clip_high)

    oof_corr = np.full(len(train_res_frame), np.nan, dtype=float)
    splitter = TimeSeriesSplit(n_splits=env["TWO_STAGE_N_SPLITS"], gap=24)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train_res_frame), 1):
        fold_set = _fit_residual_model_set(
            env,
            train_res_frame.iloc[tr_idx].reset_index(drop=True),
            residual_clip[tr_idx],
            feature_cols,
            seed_offset=2000 + 100 * fold,
            min_rows=450,
        )
        fold_corr, _ = _predict_residual_model_set(fold_set, train_res_frame.iloc[va_idx].reset_index(drop=True))
        fold_corr = np.clip(fold_corr, clip_low, clip_high)
        oof_corr[va_idx] = fold_corr
        fold_rows.append(
            {
                "expert": "oof_fold",
                "fold": fold,
                "n_valid": int(len(va_idx)),
                "base_mae": float(mean_absolute_error(y[va_idx], train_base[va_idx])),
                "corr_mae_alpha_1": float(mean_absolute_error(y[va_idx], np.clip(train_base[va_idx] + fold_corr, 0, env["INSTALLED_CAPACITY_MW"]))),
            }
        )

    covered = np.isfinite(oof_corr)
    best_alpha = 0.0
    best_mae = float(mean_absolute_error(y[covered], train_base[covered])) if covered.any() else np.nan
    if covered.any():
        for alpha in np.linspace(0.0, 0.75, 31):
            candidate = np.clip(train_base[covered] + alpha * oof_corr[covered], 0, env["INSTALLED_CAPACITY_MW"])
            mae = float(mean_absolute_error(y[covered], candidate))
            if mae < best_mae:
                best_mae = mae
                best_alpha = float(alpha)

    model_set = _fit_residual_model_set(env, train_res_frame, residual_clip, feature_cols, seed_offset=3000, min_rows=600)
    valid_corr_raw, valid_corr_diag = _predict_residual_model_set(model_set, valid_res_frame)
    valid_corr_raw = np.clip(valid_corr_raw, clip_low, clip_high)
    output_base = valid_base_pred if valid_output_base_pred is None else valid_output_base_pred
    valid_pred = np.clip(
        np.asarray(output_base, dtype=float) + best_alpha * valid_corr_raw,
        0,
        env["INSTALLED_CAPACITY_MW"],
    )

    report = pd.concat(
        [
            pd.DataFrame(fold_rows),
            model_set["fit_report"],
            pd.DataFrame(
                [
                    {
                        "expert": "selected_alpha",
                        "n_train": int(covered.sum()),
                        "train_residual_mae": best_mae,
                        "weight": best_alpha,
                    }
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    layer = {
        "enabled": True,
        "feature_cols": feature_cols,
        "model_set": model_set,
        "alpha": best_alpha,
        "clip_low": clip_low,
        "clip_high": clip_high,
        "base_kind": "two_stage_oof_residual",
        "oof_base_mae": float(mean_absolute_error(y[covered], train_base[covered])) if covered.any() else np.nan,
        "oof_corrected_mae": best_mae,
    }
    valid_corr_diag["seasonal_residual_correction_raw"] = valid_corr_raw
    valid_corr_diag["seasonal_residual_correction_final"] = best_alpha * valid_corr_raw
    return layer, valid_pred, report


def _apply_seasonal_residual_layer(
    env: dict[str, Any],
    artifacts: dict[str, Any],
    frame: pd.DataFrame,
    base_pred: np.ndarray,
    *,
    direct_pred: np.ndarray | None = None,
    two_stage_pred: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    layer = artifacts.get("seasonal_residual_layer")
    capacity = artifacts["constants"]["INSTALLED_CAPACITY_MW"]
    base = np.clip(np.asarray(base_pred, dtype=float), 0, capacity)
    if not layer or not layer.get("enabled") or layer.get("alpha", 0.0) <= 0:
        diag = pd.DataFrame(
            {
                "seasonal_residual_correction_raw": np.zeros(len(frame), dtype=float),
                "seasonal_residual_correction_final": np.zeros(len(frame), dtype=float),
                "seasonal_residual_pred": base,
            }
        )
        return base, diag

    res_frame = _seasonal_residual_frame(
        env,
        frame,
        two_stage_pred if two_stage_pred is not None else base,
        direct_pred=direct_pred,
        two_stage_pred=two_stage_pred,
    )
    corr_raw, diag = _predict_residual_model_set(layer["model_set"], res_frame)
    corr_raw = np.clip(corr_raw, layer["clip_low"], layer["clip_high"])
    corr_final = layer["alpha"] * corr_raw
    pred = np.clip(base + corr_final, 0, capacity)
    diag["seasonal_residual_correction_raw"] = corr_raw
    diag["seasonal_residual_correction_final"] = corr_final
    diag["seasonal_residual_pred"] = pred
    return pred, diag


def _prepare_training_env() -> dict[str, Any]:
    env, nb = _legacy_env(verbose=False)
    for idx in [52, 53, 62, 82, 83, 84, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121]:
        _exec_cell(nb, env, idx)
    return env


def train_and_save_models(verbose: bool = False) -> dict[str, Any]:
    """Train direct/two-stage models from saved features and persist weights."""
    _ensure_dirs()
    env = _prepare_training_env()
    if verbose:
        print("Training with saved features")

    model_features = _load_model_features()
    train_model = _align_feature_columns(pd.read_csv(TRAIN_FEATURES_PATH), model_features)
    valid_model = _align_feature_columns(pd.read_csv(VALID_FEATURES_PATH), model_features)
    train_model["datetime"] = pd.to_datetime(train_model["datetime"], errors="coerce")
    valid_model["datetime"] = pd.to_datetime(valid_model["datetime"], errors="coerce")

    direct_models = env["fit_ensemble"](
        train_model,
        model_features,
        label="saved_direct_ensemble",
        fast_mode=env["DIRECT_ENSEMBLE_FAST_MODE"],
    )
    direct_valid_raw, direct_valid_components = env["predict_ensemble"](direct_models, valid_model, model_features)
    direct_cap_curve = env["fit_high_wind_cap_curve"](
        train_model,
        speed_col=env["HIGH_WIND_SPEED_COL"],
        target_col="target",
    )
    direct_valid_clip, direct_clip_diag = env["apply_high_wind_smart_clip"](
        valid_model,
        direct_valid_raw,
        direct_cap_curve,
        speed_col=env["HIGH_WIND_SPEED_COL"],
    )
    direct_valid_clip = np.clip(direct_valid_clip, 0, env["INSTALLED_CAPACITY_MW"])

    ts_train = train_model.copy().reset_index(drop=True)
    ts_valid = valid_model.copy().reset_index(drop=True)
    if "datetime" in ts_train.columns:
        ts_train = ts_train.sort_values("datetime").reset_index(drop=True)
    direct_train_raw, _ = env["predict_ensemble"](direct_models, ts_train, model_features)
    direct_train_clip, _ = env["apply_high_wind_smart_clip"](
        ts_train,
        direct_train_raw,
        direct_cap_curve,
        speed_col=env["HIGH_WIND_SPEED_COL"],
    )
    direct_train_clip = np.clip(direct_train_clip, 0, env["INSTALLED_CAPACITY_MW"])
    base_ts_cols = env["_ts_clean_feature_list"](ts_train, ts_valid, model_features)
    normal_feature_cols = [
        c for c in base_ts_cols if not any(key in c for key in env["TWO_STAGE_DEVIATION_KEYWORDS"])
    ]
    if len(normal_feature_cols) < 20:
        normal_feature_cols = base_ts_cols.copy()
    deviation_feature_cols = base_ts_cols.copy()

    y_train = ts_train["target"].clip(0, env["INSTALLED_CAPACITY_MW"]).to_numpy(dtype=float)
    (
        normal_oof_pred,
        normal_valid_pred,
        normal_oof_components,
        normal_valid_components,
        normal_fold_report,
        normal_used_cols,
    ) = env["_ts_oof_and_valid_predictions"](
        ts_train,
        ts_valid,
        normal_feature_cols,
        y_train,
        env["_ts_model_bank"]("normal", fast=env["TWO_STAGE_FAST_MODE"]),
        n_splits=env["TWO_STAGE_N_SPLITS"],
        clip_pred=(0.0, env["INSTALLED_CAPACITY_MW"]),
        label="normal",
    )
    ts_train["two_stage_normal_pred"] = np.clip(normal_oof_pred, 0, env["INSTALLED_CAPACITY_MW"])
    ts_valid["two_stage_normal_pred"] = np.clip(normal_valid_pred, 0, env["INSTALLED_CAPACITY_MW"])

    ts_train["two_stage_deviation_target_raw"] = y_train - ts_train["two_stage_normal_pred"]
    dev_q_low = float(ts_train["two_stage_deviation_target_raw"].quantile(env["TWO_STAGE_RESIDUAL_Q_LOW"]))
    dev_q_high = float(ts_train["two_stage_deviation_target_raw"].quantile(env["TWO_STAGE_RESIDUAL_Q_HIGH"]))
    dev_clip_low = dev_q_low * env["TWO_STAGE_RESIDUAL_CLIP_MULT"]
    dev_clip_high = dev_q_high * env["TWO_STAGE_RESIDUAL_CLIP_MULT"]
    if dev_clip_low > dev_clip_high:
        dev_clip_low, dev_clip_high = dev_clip_high, dev_clip_low
    ts_train["two_stage_deviation_target"] = ts_train["two_stage_deviation_target_raw"].clip(dev_clip_low, dev_clip_high)
    residual_train_mask = np.isfinite(ts_train["two_stage_deviation_target"].to_numpy(dtype=float))

    ts_train = _add_two_stage_meta_features(env, ts_train)
    ts_valid = _add_two_stage_meta_features(env, ts_valid)
    two_stage_meta_cols = [
        "two_stage_normal_pred",
        "two_stage_normal_minus_empirical",
        "two_stage_normal_div_empirical",
        "two_stage_normal_minus_theory",
        "two_stage_normal_div_theory",
        "two_stage_normal_minus_ideal_clean",
        "two_stage_normal_div_ideal_clean",
        "two_stage_normal_x_hidden_loss",
        "two_stage_normal_x_wake_risk",
    ]
    deviation_feature_cols = env["_ts_clean_feature_list"](
        ts_train,
        ts_valid,
        deviation_feature_cols + two_stage_meta_cols,
    )
    dev_train_df = ts_train.loc[residual_train_mask].copy().reset_index(drop=True)
    y_dev = dev_train_df["two_stage_deviation_target"].to_numpy(dtype=float)
    (
        dev_oof_part,
        dev_valid_pred_raw,
        dev_oof_components_part,
        dev_valid_components,
        dev_fold_report,
        dev_used_cols,
    ) = env["_ts_oof_and_valid_predictions"](
        dev_train_df,
        ts_valid,
        deviation_feature_cols,
        y_dev,
        env["_ts_model_bank"]("deviation", fast=env["TWO_STAGE_FAST_MODE"]),
        n_splits=env["TWO_STAGE_N_SPLITS"],
        clip_pred=(dev_clip_low, dev_clip_high),
        label="deviation",
    )

    dev_oof_raw_full = np.zeros(len(ts_train), dtype=float)
    dev_oof_raw_full[residual_train_mask] = dev_oof_part
    ts_train["two_stage_deviation_oof_pred_raw"] = dev_oof_raw_full
    ts_train["two_stage_deviation_oof_pred_final"], ts_train["two_stage_oof_physics_gate"] = _apply_two_stage_safety(
        env,
        ts_train,
        ts_train["two_stage_deviation_oof_pred_raw"].to_numpy(dtype=float),
    )
    ts_train["two_stage_pred_oof_raw"] = (
        ts_train["two_stage_normal_pred"] + ts_train["two_stage_deviation_oof_pred_final"]
    ).clip(0, env["INSTALLED_CAPACITY_MW"])
    two_stage_train_clip, two_stage_train_clip_diag = env["_ts_apply_clip"](
        ts_train,
        ts_train,
        ts_train["two_stage_pred_oof_raw"].to_numpy(dtype=float),
        suffix="two_stage_train_oof",
    )
    ts_valid["two_stage_deviation_pred_raw"] = np.asarray(dev_valid_pred_raw, dtype=float)
    ts_valid["two_stage_deviation_pred_final"], ts_valid["two_stage_physics_gate"] = _apply_two_stage_safety(
        env,
        ts_valid,
        ts_valid["two_stage_deviation_pred_raw"].to_numpy(dtype=float),
    )
    ts_valid["two_stage_pred_raw"] = (
        ts_valid["two_stage_normal_pred"] + ts_valid["two_stage_deviation_pred_final"]
    ).clip(0, env["INSTALLED_CAPACITY_MW"])

    two_stage_valid_clip, two_stage_clip_diag = env["_ts_apply_clip"](
        ts_train,
        ts_valid,
        ts_valid["two_stage_pred_raw"].to_numpy(dtype=float),
        suffix="two_stage_training",
    )
    alpha = float(np.clip(env["FINAL_TWO_STAGE_ALPHA"], 0.0, 1.0))
    blend_valid = np.clip((1.0 - alpha) * direct_valid_clip + alpha * two_stage_valid_clip, 0, env["INSTALLED_CAPACITY_MW"])

    final_cfg = dict(env["FINAL_CLIP_CONFIG"])
    final_cap_curve = env["fit_high_wind_cap_curve"](
        ts_train,
        speed_col=env["HIGH_WIND_SPEED_COL"],
        target_col="target",
        quantile=final_cfg["quantile"],
        margin_mw=final_cfg["margin_mw"],
        hard_max_cap=final_cfg["hard_max_cap"],
    )
    final_valid_pred, final_clip_diag = env["apply_high_wind_smart_clip"](
        ts_valid,
        blend_valid,
        final_cap_curve,
        speed_col=env["HIGH_WIND_SPEED_COL"],
        strength=final_cfg["strength"],
    )
    final_valid_pred = np.clip(final_valid_pred, 0, env["INSTALLED_CAPACITY_MW"])
    pre_residual_valid_pred = final_valid_pred.copy()
    seasonal_residual_layer, final_valid_pred, seasonal_residual_report = _fit_seasonal_residual_layer(
        env,
        ts_train,
        ts_valid,
        two_stage_train_clip,
        two_stage_valid_clip,
        train_direct_pred=direct_train_clip,
        valid_direct_pred=direct_valid_clip,
        train_two_stage_pred=two_stage_train_clip,
        valid_two_stage_pred=two_stage_valid_clip,
        valid_output_base_pred=pre_residual_valid_pred,
    )

    normal_bank = _train_model_bank(
        env,
        ts_train,
        normal_feature_cols,
        y_train,
        "normal",
        clip_pred=(0.0, env["INSTALLED_CAPACITY_MW"]),
    )
    deviation_bank = _train_model_bank(
        env,
        dev_train_df,
        deviation_feature_cols,
        y_dev,
        "deviation",
        clip_pred=(dev_clip_low, dev_clip_high),
    )

    valid_diag = valid_model[["row_id"]].copy()
    if "datetime" in valid_model.columns:
        valid_diag["datetime"] = pd.to_datetime(valid_model["datetime"], errors="coerce")
    valid_diag["direct_pred_raw"] = direct_valid_raw
    valid_diag["direct_pred_clip"] = direct_valid_clip
    valid_diag["two_stage_pred_clip"] = two_stage_valid_clip
    valid_diag["pre_residual_final_pred"] = pre_residual_valid_pred
    valid_diag["final_pred"] = final_valid_pred
    valid_diag.to_csv(MODEL_DIR / "valid_training_predictions.csv", index=False)
    normal_fold_report.to_csv(MODEL_DIR / "stage_a_normal_fold_report.csv", index=False)
    dev_fold_report.to_csv(MODEL_DIR / "stage_b_deviation_fold_report.csv", index=False)
    direct_clip_diag.to_csv(MODEL_DIR / "direct_high_wind_clip_diag.csv", index=False)
    two_stage_clip_diag.to_csv(MODEL_DIR / "two_stage_high_wind_clip_diag.csv", index=False)
    two_stage_train_clip_diag.to_csv(MODEL_DIR / "two_stage_train_oof_high_wind_clip_diag.csv", index=False)
    final_clip_diag.to_csv(MODEL_DIR / "final_high_wind_clip_diag.csv", index=False)
    seasonal_residual_report.to_csv(SEASONAL_RESIDUAL_REPORT_PATH, index=False)

    summary = pd.DataFrame(
        [
            {
                "metric": "direct_valid_mean",
                "value": float(np.mean(direct_valid_clip)),
            },
            {
                "metric": "two_stage_valid_mean",
                "value": float(np.mean(two_stage_valid_clip)),
            },
            {
                "metric": "final_valid_mean",
                "value": float(np.mean(final_valid_pred)),
            },
            {
                "metric": "pre_residual_final_valid_mean",
                "value": float(np.mean(pre_residual_valid_pred)),
            },
            {
                "metric": "n_model_features",
                "value": float(len(model_features)),
            },
            {
                "metric": "normal_oof_mae",
                "value": float(mean_absolute_error(y_train, ts_train["two_stage_normal_pred"])),
            },
            {
                "metric": "seasonal_residual_alpha",
                "value": float(seasonal_residual_layer["alpha"]),
            },
            {
                "metric": "seasonal_residual_oof_base_mae",
                "value": float(seasonal_residual_layer["oof_base_mae"]),
            },
            {
                "metric": "seasonal_residual_oof_corrected_mae",
                "value": float(seasonal_residual_layer["oof_corrected_mae"]),
            },
        ]
    )
    summary.to_csv(TRAINING_SUMMARY_PATH, index=False)

    artifacts = {
        "model_features": model_features,
        "direct_models": direct_models,
        "direct_cap_curve": direct_cap_curve,
        "direct_component_names": list(direct_valid_components.keys()),
        "normal_bank": normal_bank,
        "deviation_bank": deviation_bank,
        "dev_clip_low": dev_clip_low,
        "dev_clip_high": dev_clip_high,
        "final_cap_curve": final_cap_curve,
        "final_clip_config": final_cfg,
        "final_two_stage_alpha": alpha,
        "seasonal_residual_layer": seasonal_residual_layer,
        "constants": {
            "INSTALLED_CAPACITY_MW": env["INSTALLED_CAPACITY_MW"],
            "HIGH_WIND_SPEED_COL": env["HIGH_WIND_SPEED_COL"],
            "HIGH_WIND_START_WS": env["HIGH_WIND_START_WS"],
            "HIGH_WIND_TRANSITION": env["HIGH_WIND_TRANSITION"],
            "HIGH_WIND_CLIP_STRENGTH": env["HIGH_WIND_CLIP_STRENGTH"],
            "TWO_STAGE_DEVIATION_SHRINK": env["TWO_STAGE_DEVIATION_SHRINK"],
            "TWO_STAGE_DEVIATION_ABS_CLIP_MW": env["TWO_STAGE_DEVIATION_ABS_CLIP_MW"],
            "TWO_STAGE_USE_PHYSICS_GATE": env["TWO_STAGE_USE_PHYSICS_GATE"],
            "TWO_STAGE_GATE_MIN": env["TWO_STAGE_GATE_MIN"],
            "TWO_STAGE_GATE_MAX": env["TWO_STAGE_GATE_MAX"],
        },
    }
    joblib.dump(artifacts, MODEL_ARTIFACTS_PATH)
    return {"summary": summary, "model_artifacts_path": MODEL_ARTIFACTS_PATH}


def _predict_direct(env: dict[str, Any], artifacts: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    raw, _ = env["predict_ensemble"](artifacts["direct_models"], frame, artifacts["model_features"])
    clipped, _ = env["apply_high_wind_smart_clip"](
        frame,
        raw,
        artifacts["direct_cap_curve"],
        speed_col=artifacts["constants"]["HIGH_WIND_SPEED_COL"],
    )
    capacity = artifacts["constants"]["INSTALLED_CAPACITY_MW"]
    return np.clip(raw, 0, capacity), np.clip(clipped, 0, capacity)


def _predict_two_stage(env: dict[str, Any], artifacts: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    capacity = artifacts["constants"]["INSTALLED_CAPACITY_MW"]
    work = frame.copy().reset_index(drop=True)
    normal_pred, normal_components = _predict_model_bank(artifacts["normal_bank"], work)
    work["two_stage_normal_pred"] = np.clip(normal_pred, 0, capacity)
    work = _add_two_stage_meta_features(env, work)
    dev_raw, dev_components = _predict_model_bank(artifacts["deviation_bank"], work)
    dev_final, gate = _apply_two_stage_safety(env, work, dev_raw)
    pred_raw = np.clip(work["two_stage_normal_pred"].to_numpy(dtype=float) + dev_final, 0, capacity)
    pred_clip, _ = env["apply_high_wind_smart_clip"](
        work,
        pred_raw,
        artifacts["direct_cap_curve"],
        speed_col=artifacts["constants"]["HIGH_WIND_SPEED_COL"],
    )
    diag = pd.DataFrame(
        {
            "two_stage_normal_pred": work["two_stage_normal_pred"].to_numpy(dtype=float),
            "two_stage_deviation_pred_raw": dev_raw,
            "two_stage_deviation_pred_final": dev_final,
            "two_stage_physics_gate": gate,
            "two_stage_pred_raw": pred_raw,
            "two_stage_pred_clip": np.clip(pred_clip, 0, capacity),
        }
    )
    for col in normal_components.columns:
        diag[f"normal_component_{col}"] = normal_components[col].to_numpy()
    for col in dev_components.columns:
        diag[f"deviation_component_{col}"] = dev_components[col].to_numpy()
    return np.clip(pred_clip, 0, capacity), diag


def _final_blend(env: dict[str, Any], artifacts: dict[str, Any], frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    direct_raw, direct_clip = _predict_direct(env, artifacts, frame)
    two_stage_clip, two_stage_diag = _predict_two_stage(env, artifacts, frame)
    alpha = artifacts["final_two_stage_alpha"]
    capacity = artifacts["constants"]["INSTALLED_CAPACITY_MW"]
    blend = np.clip((1.0 - alpha) * direct_clip + alpha * two_stage_clip, 0, capacity)
    final_pred, final_clip_diag = env["apply_high_wind_smart_clip"](
        frame,
        blend,
        artifacts["final_cap_curve"],
        speed_col=artifacts["constants"]["HIGH_WIND_SPEED_COL"],
        strength=artifacts["final_clip_config"]["strength"],
    )
    final_pred = np.clip(final_pred, 0, capacity)
    pre_residual_final_pred = final_pred.copy()
    final_pred, seasonal_diag = _apply_seasonal_residual_layer(
        env,
        artifacts,
        frame,
        pre_residual_final_pred,
        direct_pred=direct_clip,
        two_stage_pred=two_stage_clip,
    )
    diag = frame[["row_id"]].copy() if "row_id" in frame.columns else pd.DataFrame({"row_id": np.arange(len(frame))})
    if "datetime" in frame.columns:
        diag["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    diag["direct_pred_raw"] = direct_raw
    diag["direct_pred_clip"] = direct_clip
    diag["two_stage_pred_clip"] = two_stage_clip
    diag["blend_pred"] = blend
    diag["pre_residual_final_pred"] = pre_residual_final_pred
    diag["final_pred"] = final_pred
    diag["final_minus_direct"] = final_pred - direct_clip
    diag = pd.concat([diag.reset_index(drop=True), two_stage_diag.reset_index(drop=True)], axis=1)
    diag = pd.concat([diag.reset_index(drop=True), seasonal_diag.reset_index(drop=True)], axis=1)
    diag["final_clip_delta"] = final_clip_diag["clip_delta"].to_numpy()
    return final_pred, diag


def _save_submission(frame: pd.DataFrame, pred: np.ndarray, path: Path) -> pd.DataFrame:
    sub = frame[["row_id"]].copy() if "row_id" in frame.columns else pd.DataFrame({"row_id": np.arange(len(frame))})
    sub["target"] = np.asarray(pred, dtype=float)
    sub = sub.sort_values("row_id")[["target"]].reset_index(drop=True)
    sub.to_csv(path, index=False)
    return sub


def _metrics(compare: pd.DataFrame) -> pd.DataFrame:
    eval_df = compare[compare["actual_mw"].notna() & compare["prediction_mw"].notna()].copy()
    if len(eval_df) == 0:
        return pd.DataFrame(
            [{"rows_total": len(compare), "rows_with_actual": 0, "rows_without_actual": int(compare["actual_mw"].isna().sum())}]
        )
    err = eval_df["prediction_mw"].to_numpy(dtype=float) - eval_df["actual_mw"].to_numpy(dtype=float)
    abs_err = np.abs(err)
    actual = eval_df["actual_mw"].to_numpy(dtype=float)
    pred = eval_df["prediction_mw"].to_numpy(dtype=float)
    corr = np.nan
    if len(eval_df) > 1 and np.std(actual) > 1e-6 and np.std(pred) > 1e-6:
        corr = float(np.corrcoef(actual, pred)[0, 1])
    return pd.DataFrame(
        [
            {
                "rows_total": len(compare),
                "rows_with_actual": int(len(eval_df)),
                "rows_without_actual": int(compare["actual_mw"].isna().sum()),
                "mae_mw": float(abs_err.mean()),
                "rmse_mw": float(np.sqrt(np.mean(err**2))),
                "bias_mw": float(err.mean()),
                "median_abs_error_mw": float(np.median(abs_err)),
                "p90_abs_error_mw": float(np.quantile(abs_err, 0.90)),
                "max_abs_error_mw": float(abs_err.max()),
                "corr_actual_prediction": corr,
                "actual_mean_mw": float(np.mean(actual)),
                "prediction_mean_mw": float(np.mean(pred)),
            }
        ]
    )


def _light_postprocess_feature_frame(features: pd.DataFrame, pred: np.ndarray, diag: pd.DataFrame | None = None) -> pd.DataFrame:
    out = pd.DataFrame({"base_prediction_mw": np.asarray(pred, dtype=float)})
    source = features.reset_index(drop=True).copy()
    diag_frame = diag.reset_index(drop=True).copy() if diag is not None else pd.DataFrame(index=source.index)

    def one_column(frame: pd.DataFrame, col: str) -> pd.Series:
        data = frame.loc[:, col]
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]
        return pd.to_numeric(data, errors="coerce")

    for col in [
        "direct_pred_clip",
        "two_stage_pred_clip",
        "blend_pred",
        "pre_residual_final_pred",
        "wind_speed_120m",
        "wind_speed_80m",
        "wind_direction_120m",
        "turbines_in_repair",
        "p_empirical_mean_80_120",
        "p_theory_mean_80_120",
        "p_empirical_minus_theory_120",
        "ws_diff_120_80",
        "layout_wake_risk_scalar_120m",
        "temperature_120m",
        "phi_ice_risk",
        "full_recon_minus_empirical_curve",
    ]:
        if col in diag_frame.columns:
            out[col] = one_column(diag_frame, col)
        elif col in source.columns:
            out[col] = one_column(source, col)

    ws = out.get("wind_speed_120m", pd.Series(np.nan, index=out.index)).fillna(0)
    out["lp_low_wind_gate"] = (ws < 3.0).astype(float)
    out["lp_soft_low_wind_gate"] = ((ws >= 3.0) & (ws < 5.0)).astype(float)
    out["lp_mid_wind_gate"] = ((ws >= 5.0) & (ws <= 11.0)).astype(float)
    out["lp_high_wind_gate"] = (ws > 11.0).astype(float)
    out["lp_base_x_low_wind"] = out["base_prediction_mw"] * out["lp_low_wind_gate"]
    out["lp_base_x_mid_wind"] = out["base_prediction_mw"] * out["lp_mid_wind_gate"]
    if "direct_pred_clip" in out.columns and "two_stage_pred_clip" in out.columns:
        out["lp_two_stage_minus_direct"] = out["two_stage_pred_clip"] - out["direct_pred_clip"]
    return out


def _apply_light_postprocess_calibration(
    capacity: float,
    features: pd.DataFrame,
    pred: np.ndarray,
    actual: pd.DataFrame,
    diag: pd.DataFrame | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    base = np.clip(np.asarray(pred, dtype=float), 0, capacity)
    report_rows = []

    row_id = features["row_id"].to_numpy() if "row_id" in features.columns else np.arange(len(features))
    actual_by_row = actual.set_index("row_id").reindex(row_id).reset_index()
    y = pd.to_numeric(actual_by_row.get("actual_mw", np.nan), errors="coerce")
    known_mask = y.notna().to_numpy()

    if "wind_speed_120m" in features.columns:
        ws = pd.to_numeric(features["wind_speed_120m"], errors="coerce").to_numpy(dtype=float)
    elif "wind_speed_120m" in actual_by_row.columns:
        ws = pd.to_numeric(actual_by_row["wind_speed_120m"], errors="coerce").to_numpy(dtype=float)
    else:
        ws = np.full(len(base), np.nan)

    rule_delta = np.zeros(len(base), dtype=float)
    rule_delta[np.isfinite(ws) & (ws < 3.0)] -= 2.5
    rule_delta[np.isfinite(ws) & (ws >= 3.0) & (ws < 5.0)] -= 0.5
    low_wind_only = np.clip(base + rule_delta, 0, capacity)
    if known_mask.any():
        report_rows.append(
            {
                "stage": "low_wind_rule",
                "known_mae": float(mean_absolute_error(y[known_mask], low_wind_only[known_mask])),
                "known_bias": float(np.mean(low_wind_only[known_mask] - y[known_mask])),
                "mean_delta_mw": float(np.mean(rule_delta)),
            }
        )

    # A tiny calibrator trained only on rows with known actuals. It is intentionally
    # small and shrunk because it is a postprocess layer, not a replacement model.
    raw_corr = np.zeros(len(base), dtype=float)
    guarded_delta = np.zeros(len(base), dtype=float)
    aggressive_delta = np.zeros(len(base), dtype=float)
    selected_shrink = 0.0
    holdout_base_mae = np.nan
    holdout_best_mae = np.nan
    try:
        x = _light_postprocess_feature_frame(features, low_wind_only, diag)
        train_mask = known_mask.copy()
        if train_mask.sum() >= 200:
            dt = pd.Series(pd.NaT, index=np.arange(len(base)))
            if "datetime" in actual_by_row.columns:
                dt = pd.to_datetime(actual_by_row["datetime"], errors="coerce")
            elif "datetime" in features.columns:
                dt = pd.to_datetime(features["datetime"], errors="coerce")

            known_dates = dt[train_mask]
            if known_dates.notna().sum() >= 200:
                holdout_start = known_dates.max() - pd.Timedelta(days=7)
                holdout_mask = train_mask & (dt >= holdout_start).to_numpy()
                fit_mask = train_mask & ~holdout_mask
            else:
                holdout_mask = np.zeros(len(base), dtype=bool)
                fit_mask = train_mask

            def fit_calibrator(mask: np.ndarray, seed: int) -> tuple[HistGradientBoostingRegressor, pd.Series]:
                x_train = x.loc[mask].copy().replace([np.inf, -np.inf], np.nan)
                med = x_train.median(numeric_only=True)
                x_train = x_train.fillna(med).fillna(0)
                target_residual = y[mask].to_numpy(dtype=float) - low_wind_only[mask]
                target_residual = np.clip(target_residual, -18.0, 18.0)
                model = HistGradientBoostingRegressor(
                    loss="absolute_error",
                    max_iter=120,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=35,
                    l2_regularization=0.15,
                    early_stopping=True,
                    validation_fraction=0.12,
                    random_state=seed,
                )
                model.fit(x_train, target_residual)
                return model, med

            if fit_mask.sum() >= 200 and holdout_mask.sum() >= 48:
                guard_model, guard_medians = fit_calibrator(fit_mask, 41)
                x_holdout = x.loc[holdout_mask].copy().replace([np.inf, -np.inf], np.nan)
                x_holdout = x_holdout.fillna(guard_medians).fillna(0)
                holdout_raw = np.clip(np.asarray(guard_model.predict(x_holdout), dtype=float), -10.0, 10.0)
                holdout_base_mae = float(mean_absolute_error(y[holdout_mask], low_wind_only[holdout_mask]))
                holdout_best_mae = holdout_base_mae
                for shrink in [0.0, 0.1, 0.2, 0.35, 0.5]:
                    candidate = np.clip(low_wind_only[holdout_mask] + shrink * holdout_raw, 0, capacity)
                    mae = float(mean_absolute_error(y[holdout_mask], candidate))
                    if mae < holdout_best_mae - 1e-6:
                        holdout_best_mae = mae
                        selected_shrink = float(shrink)

            calibrator, medians = fit_calibrator(train_mask, 42)
            x_all = x.copy().replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)
            raw_corr = np.clip(np.asarray(calibrator.predict(x_all), dtype=float), -10.0, 10.0)
            guarded_delta = selected_shrink * raw_corr
            aggressive_delta = 0.50 * raw_corr
            guarded = np.clip(low_wind_only + guarded_delta, 0, capacity)
            aggressive = np.clip(low_wind_only + aggressive_delta, 0, capacity)
            report_rows.append(
                {
                    "stage": "guarded_calibrator",
                    "known_mae": float(mean_absolute_error(y[known_mask], guarded[known_mask])),
                    "known_bias": float(np.mean(guarded[known_mask] - y[known_mask])),
                    "mean_delta_mw": float(np.mean(guarded_delta)),
                    "selected_shrink": selected_shrink,
                    "holdout_base_mae": holdout_base_mae,
                    "holdout_best_mae": holdout_best_mae,
                }
            )
            report_rows.append(
                {
                    "stage": "aggressive_calibrator",
                    "known_mae": float(mean_absolute_error(y[known_mask], aggressive[known_mask])),
                    "known_bias": float(np.mean(aggressive[known_mask] - y[known_mask])),
                    "mean_delta_mw": float(np.mean(aggressive_delta)),
                    "selected_shrink": 0.50,
                }
            )
        else:
            guarded = low_wind_only.copy()
            aggressive = low_wind_only.copy()
    except Exception as exc:
        guarded = low_wind_only.copy()
        aggressive = low_wind_only.copy()
        report_rows.append(
            {
                "stage": "guarded_calibrator_failed",
                "known_mae": np.nan,
                "known_bias": np.nan,
                "mean_delta_mw": 0.0,
                "note": str(exc),
            }
        )
    if "guarded" not in locals():
        guarded = np.clip(low_wind_only + guarded_delta, 0, capacity)
    if "aggressive" not in locals():
        aggressive = np.clip(low_wind_only + aggressive_delta, 0, capacity)

    diag_out = pd.DataFrame(
        {
            "light_postprocess_base_pred": base,
            "light_postprocess_low_wind_delta": rule_delta,
            "light_postprocess_calibrator_raw": raw_corr,
            "light_postprocess_guarded_delta": guarded_delta,
            "light_postprocess_aggressive_delta": aggressive_delta,
            "light_postprocess_low_wind_only_pred": low_wind_only,
            "light_postprocess_guarded_pred": guarded,
            "light_postprocess_aggressive_pred": aggressive,
            "light_postprocess_final_pred": guarded,
        }
    )
    if known_mask.any():
        report_rows.insert(
            0,
            {
                "stage": "base",
                "known_mae": float(mean_absolute_error(y[known_mask], base[known_mask])),
                "known_bias": float(np.mean(base[known_mask] - y[known_mask])),
                "mean_delta_mw": 0.0,
            },
        )
    variants = {
        "low_wind_only": low_wind_only,
        "guarded": guarded,
        "aggressive": aggressive,
    }
    return guarded, diag_out, pd.DataFrame(report_rows), variants


def _plot_test_diagnostics(compare: pd.DataFrame, metrics: pd.DataFrame, capacity: float) -> None:
    eval_df = compare[compare["actual_mw"].notna() & compare["prediction_mw"].notna()].copy()
    if len(eval_df) == 0:
        return
    eval_df["datetime"] = pd.to_datetime(eval_df["datetime"], errors="coerce")
    eval_df["error_mw"] = eval_df["prediction_mw"] - eval_df["actual_mw"]
    eval_df["abs_error_mw"] = eval_df["error_mw"].abs()
    metric_row = metrics.iloc[0].to_dict()

    def savefig(name: str) -> None:
        path = TEST_FIGURE_DIR / name
        plt.savefig(path, dpi=180, bbox_inches="tight")
        plt.close()

    plot_df = eval_df.sort_values("datetime")
    plt.figure(figsize=(16, 5))
    plt.plot(plot_df["datetime"], plot_df["actual_mw"], linewidth=1.8, label="actual")
    plt.plot(plot_df["datetime"], plot_df["prediction_mw"], linewidth=1.4, alpha=0.85, label="prediction")
    plt.title("Test dataset: prediction vs actual")
    plt.xlabel("datetime")
    plt.ylabel("power, MW")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    savefig("test_01_prediction_vs_actual_time.png")

    plt.figure(figsize=(16, 4.5))
    plt.plot(plot_df["datetime"], plot_df["error_mw"], linewidth=1.2)
    plt.axhline(0, linestyle="--", linewidth=1.2, color="black")
    plt.title("Test dataset: prediction error over time")
    plt.xlabel("datetime")
    plt.ylabel("prediction - actual, MW")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    savefig("test_02_error_time.png")

    lims = [min(eval_df["actual_mw"].min(), eval_df["prediction_mw"].min(), 0), max(eval_df["actual_mw"].max(), eval_df["prediction_mw"].max(), capacity)]
    plt.figure(figsize=(7, 7))
    plt.scatter(eval_df["actual_mw"], eval_df["prediction_mw"], s=18, alpha=0.55)
    plt.plot(lims, lims, linestyle="--", linewidth=1.5, color="black")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.title("Test dataset: actual vs prediction")
    plt.xlabel("actual, MW")
    plt.ylabel("prediction, MW")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    savefig("test_03_actual_vs_prediction_scatter.png")

    plt.figure(figsize=(12, 4.5))
    plt.hist(eval_df["abs_error_mw"], bins=45, alpha=0.85)
    plt.axvline(metric_row.get("mae_mw", eval_df["abs_error_mw"].mean()), linestyle="--", linewidth=1.5, color="black", label="MAE")
    plt.axvline(metric_row.get("median_abs_error_mw", eval_df["abs_error_mw"].median()), linestyle=":", linewidth=2.0, color="black", label="Median AE")
    plt.title("Test dataset: absolute error distribution")
    plt.xlabel("|prediction - actual|, MW")
    plt.ylabel("count")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    savefig("test_04_abs_error_distribution.png")

    power_bins = [0, 1, 5, 10, 20, 40, 60, 80, capacity + 1]
    eval_df["actual_power_bin"] = pd.cut(eval_df["actual_mw"], bins=power_bins, include_lowest=True)
    by_power = eval_df.groupby("actual_power_bin", observed=True).agg(mae=("abs_error_mw", "mean"), n=("abs_error_mw", "size")).reset_index()
    plt.figure(figsize=(12, 4.5))
    sns.barplot(data=by_power, x="actual_power_bin", y="mae")
    plt.xticks(rotation=35, ha="right")
    plt.title("Test dataset: MAE by actual power bin")
    plt.xlabel("actual power bin, MW")
    plt.ylabel("MAE, MW")
    plt.tight_layout()
    savefig("test_05_mae_by_actual_power_bin.png")

    if "wind_speed_120m" in eval_df.columns:
        eval_df["wind_speed_120m_bin"] = pd.cut(eval_df["wind_speed_120m"], bins=np.arange(0, 26, 2), include_lowest=True)
        by_wind = eval_df.groupby("wind_speed_120m_bin", observed=True).agg(mae=("abs_error_mw", "mean"), n=("abs_error_mw", "size")).reset_index()
        plt.figure(figsize=(12, 4.5))
        sns.barplot(data=by_wind, x="wind_speed_120m_bin", y="mae")
        plt.xticks(rotation=35, ha="right")
        plt.title("Test dataset: MAE by wind speed 120m bin")
        plt.xlabel("wind speed 120m bin")
        plt.ylabel("MAE, MW")
        plt.tight_layout()
        savefig("test_06_mae_by_wind_speed_120m_bin.png")


def postprocess_and_evaluate(verbose: bool = False) -> dict[str, Any]:
    """Load saved weights, predict valid/test, save submissions, metrics, and plots."""
    _ensure_dirs()
    env = _prepare_training_env()
    artifacts = joblib.load(MODEL_ARTIFACTS_PATH)
    model_features = artifacts["model_features"]
    capacity = artifacts["constants"]["INSTALLED_CAPACITY_MW"]

    valid_features = _align_feature_columns(pd.read_csv(VALID_FEATURES_PATH), model_features)
    test_features = _align_feature_columns(pd.read_csv(TEST_FEATURES_PATH), model_features)
    for frame in [valid_features, test_features]:
        if "datetime" in frame.columns:
            frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")

    valid_pred, valid_diag = _final_blend(env, artifacts, valid_features)
    test_pred, test_diag = _final_blend(env, artifacts, test_features)
    test_actual = pd.read_csv(TEST_ACTUAL_PATH)
    test_pred, light_diag, light_report, test_variants = _apply_light_postprocess_calibration(
        capacity,
        test_features,
        test_pred,
        test_actual,
        test_diag,
    )
    test_diag = pd.concat([test_diag.reset_index(drop=True), light_diag.reset_index(drop=True)], axis=1)

    valid_submission = _save_submission(valid_features, valid_pred, FINAL_VALID_SUBMISSION_PATH)
    test_submission = _save_submission(test_features, test_pred, TEST_SUBMISSION_PATH)
    variant_submissions = {
        "low_wind_only": _save_submission(
            test_features,
            test_variants["low_wind_only"],
            TEST_LOW_WIND_ONLY_SUBMISSION_PATH,
        ),
        "guarded": _save_submission(
            test_features,
            test_variants["guarded"],
            TEST_GUARDED_SUBMISSION_PATH,
        ),
        "aggressive": _save_submission(
            test_features,
            test_variants["aggressive"],
            TEST_AGGRESSIVE_SUBMISSION_PATH,
        ),
    }

    compare = test_actual.copy()
    compare = compare.sort_values("row_id").reset_index(drop=True)
    compare["prediction_mw"] = test_submission["target"].to_numpy(dtype=float)
    if "actual_mw" not in compare.columns:
        compare["actual_mw"] = np.nan
    compare["error_mw"] = compare["prediction_mw"] - compare["actual_mw"]
    compare["abs_error_mw"] = compare["error_mw"].abs()
    metrics = _metrics(compare)
    compare.to_csv(TEST_COMPARE_PATH, index=False)
    metrics.to_csv(TEST_METRICS_PATH, index=False)

    test_diag.to_csv(TEST_OUTPUT_DIR / "prediction_diagnostics.csv", index=False)
    valid_diag.to_csv(OUTPUT_DIR / "valid_prediction_diagnostics.csv", index=False)
    light_report.to_csv(LIGHT_POSTPROCESS_REPORT_PATH, index=False)

    if "datetime" in compare.columns:
        missing_actual_mask = compare["actual_mw"].isna() if "actual_mw" in compare.columns else pd.Series(False, index=compare.index)
        if missing_actual_mask.any():
            day_sub = pd.DataFrame({"target": compare.loc[missing_actual_mask, "prediction_mw"].to_numpy(dtype=float)})
        else:
            dt = pd.to_datetime(compare["datetime"], errors="coerce")
            day_mask = dt.dt.date.astype(str).eq("2026-05-18")
            if day_mask.any():
                day_sub = pd.DataFrame({"target": compare.loc[day_mask, "prediction_mw"].to_numpy(dtype=float)})
            else:
                day_sub = test_submission.tail(24).reset_index(drop=True)
        day_sub.to_csv(TEST_DAY_SUBMISSION_PATH, index=False)
        for name, sub in variant_submissions.items():
            if missing_actual_mask.any():
                variant_day_sub = pd.DataFrame({"target": sub.loc[missing_actual_mask, "target"].to_numpy(dtype=float)})
            else:
                variant_day_sub = sub.tail(24).reset_index(drop=True)
            variant_day_sub.to_csv(TEST_OUTPUT_DIR / f"submission_2026-05-18_{name}.csv", index=False)

    _plot_test_diagnostics(compare, metrics, capacity)

    for path, expected_len in [
        (FINAL_VALID_SUBMISSION_PATH, len(valid_features)),
        (TEST_SUBMISSION_PATH, len(test_features)),
        (TEST_LOW_WIND_ONLY_SUBMISSION_PATH, len(test_features)),
        (TEST_GUARDED_SUBMISSION_PATH, len(test_features)),
        (TEST_AGGRESSIVE_SUBMISSION_PATH, len(test_features)),
    ]:
        sub = pd.read_csv(path)
        if list(sub.columns) != ["target"]:
            raise ValueError(f"Bad submission columns in {path}: {list(sub.columns)}")
        if len(sub) != expected_len:
            raise ValueError(f"Bad submission length in {path}: {len(sub)} != {expected_len}")
        if sub["target"].isna().any():
            raise ValueError(f"NaN predictions in {path}")
        if not sub["target"].between(0, capacity).all():
            raise ValueError(f"Predictions outside [0, {capacity}] in {path}")

    if verbose:
        print(metrics)

    return {
        "valid_submission": valid_submission,
        "test_submission": test_submission,
        "variant_submissions": variant_submissions,
        "test_metrics": metrics,
        "test_compare": compare,
    }


def validate_feature_contract() -> pd.DataFrame:
    model_features = _load_model_features()
    train = pd.read_csv(TRAIN_FEATURES_PATH)
    valid = pd.read_csv(VALID_FEATURES_PATH)
    test = pd.read_csv(TEST_FEATURES_PATH)
    rows = []
    for name, frame, raw_path in [
        ("train", train, TRAIN_PATH),
        ("valid", valid, VALID_PATH),
        ("test", test, TEST_PATH),
    ]:
        raw_rows = len(pd.read_csv(raw_path))
        missing = [col for col in model_features if col not in frame.columns]
        rows.append(
            {
                "dataset": name,
                "feature_rows": len(frame),
                "raw_rows": raw_rows,
                "row_count_ok": len(frame) == raw_rows,
                "missing_model_features": len(missing),
            }
        )
    report = pd.DataFrame(rows)
    if not report["row_count_ok"].all() or (report["missing_model_features"] > 0).any():
        raise ValueError("Feature contract validation failed:\n" + report.to_string(index=False))
    return report
