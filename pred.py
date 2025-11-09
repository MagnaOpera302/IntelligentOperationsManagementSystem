# train_timeseries_per_category.py
# Most stable at Python 3.12.0 

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt

try:
    import xgboost as xgb
    import joblib
except ImportError as e:
    raise SystemExit("Please install xgboost: pip install xgboost") from e

# ----------------------------
# Config
# ----------------------------
INPUT_CSV = "CleanedUpData.csv"
OUTPUT_PROCESSED = "processed_for_ml.csv"
OUTPUT_PREDICTIONS = "predictions_testset.csv"
MODEL_XGB = "model.xgb"
MODEL_PKL = "model.pkl"

# Feature settings
LAGS = [1, 2, 3]
ROLL_WINDOWS = [3, 6]           # will auto-trim if your data is short
MAX_TEST_MONTHS = 6             # cap test horizon to keep a reasonable holdout
TEST_FRACTION = 0.2             # ~20% of months as test if longer series
DAYFIRST = False                # set to False if your dates are YYYY-MM-DD
RANDOM_SEED = 42

# ----------------------------
# Helpers
# ----------------------------
def find_date_column(df: pd.DataFrame) -> str:
    """Try a few likely date column names; fall back to heuristic search."""
    candidates = ["Year-Month", "YearMonth", "Year_Month", "Date", "Order Date", "OrderDate"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == "O":
            s = df[c].astype(str)
            frac_datey = s.str.contains(r"\d{2,4}[-/]\d{1,2}", regex=True, na=False).mean()
            if frac_datey > 0.3:
                return c
    raise ValueError("Could not locate a date column.")

def coerce_sales_to_numeric(df: pd.DataFrame, sales_col: str = "Sales") -> pd.DataFrame:
    if sales_col not in df.columns:
        for alt in ["Qty", "Quantity", "Units", "Sales_Units", "sales"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "Sales"})
                break
    if "Sales" not in df.columns:
        raise ValueError("No 'Sales' column found.")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Sales"])
    return df

def ensure_category_column(df: pd.DataFrame, cat_col: str = "Category") -> pd.DataFrame:
    if cat_col not in df.columns:
        for alt in ["Cat", "Product Category", "ItemCategory", "Segment"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "Category"})
                break
    if "Category" not in df.columns:
        raise ValueError("No 'Category' column found.")
    df["Category"] = df["Category"].astype(str)
    return df

def monthly_aggregate(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df["_raw_date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=DAYFIRST, infer_datetime_format=True)
    df = df.dropna(subset=["_raw_date"])
    df["Period"] = df["_raw_date"].dt.to_period("M").dt.to_timestamp()
    agg = df.groupby(["Category", "Period"], as_index=False)["Sales"].sum().sort_values(["Category", "Period"]).reset_index(drop=True)
    return agg

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = df["Period"].dt.year
    df["Month"] = df["Period"].dt.month

    min_len = df.groupby("Category").size().min()
    effective_rolls = [w for w in ROLL_WINDOWS if w <= max(2, (min_len - 1))]
    if not effective_rolls:
        effective_rolls = [2]

    for lag in LAGS:
        df[f"lag_{lag}"] = df.groupby("Category")["Sales"].shift(lag)

    for w in effective_rolls:
        df[f"roll_{w}"] = df.groupby("Category")["Sales"].shift(1).rolling(window=w, min_periods=w).mean()

    feature_cols = ["Year", "Month"] + [f"lag_{l}" for l in LAGS] + [f"roll_{w}" for w in effective_rolls]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df, feature_cols

def time_aware_split(df: pd.DataFrame, test_fraction=TEST_FRACTION, max_test_months=MAX_TEST_MONTHS):
    months = np.array(sorted(df["Period"].unique()))
    if len(months) == 0:
        raise ValueError("No months available after feature engineering.")
    n_test = max(1, int(round(len(months) * test_fraction)))
    n_test = min(max_test_months, n_test)
    test_months = months[-n_test:]
    train = df[~df["Period"].isin(test_months)].copy()
    test = df[df["Period"].isin(test_months)].copy()
    print(f"[Split] Months total: {len(months)} | Test months: {len(test_months)} ({pd.to_datetime(test_months[0]).date()} .. {pd.to_datetime(test_months[-1]).date()})")
    return train, test, test_months

def main():
    if not Path(INPUT_CSV).exists():
        raise SystemExit(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    print("[Loaded]", df.shape, "columns:", list(df.columns))

    df = ensure_category_column(df)
    df = coerce_sales_to_numeric(df)

    date_col = find_date_column(df)
    print(f"[Info] Detected date column: '{date_col}'")

    dfm = monthly_aggregate(df, date_col)
    print("[Monthly aggregated] shape:", dfm.shape)
    print(dfm.head())

    overall = dfm.groupby("Period", as_index=False)["Sales"].sum()
    ax = overall.plot(x="Period", y="Sales", title="Overall Monthly Sales", legend=False)
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    plt.tight_layout()
    plt.show()

    top_cats = dfm["Category"].value_counts().index.tolist()
    N = min(6, len(top_cats))
    for cat in top_cats[:N]:
        tmp = dfm[dfm["Category"] == cat]
        ax = tmp.plot(x="Period", y="Sales", title=f"Monthly Sales — {cat}", legend=False)
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")
        plt.tight_layout()
        plt.show()

    df_feat, feature_cols = build_features(dfm)
    df_feat["Category_raw"] = df_feat["Category"]
    df_feat = pd.get_dummies(df_feat, columns=["Category"], drop_first=True)

    if "_raw_date" in df_feat.columns:
        df_feat = df_feat.drop(columns=["_raw_date"])

    non_numeric = df_feat.select_dtypes(exclude=[np.number, bool]).columns.tolist()
    non_numeric_to_drop = [c for c in non_numeric if c not in ["Category_raw", "Period"]]
    if non_numeric_to_drop:
        print("[Warn] Non-numeric columns will be dropped for modeling:", non_numeric_to_drop)
        df_feat = df_feat.drop(columns=non_numeric_to_drop)

    df_feat.to_csv(OUTPUT_PROCESSED, index=False)
    print(f"[Saved] Processed ML table -> {OUTPUT_PROCESSED}")

    df_for_split = monthly_aggregate(df, date_col)
    df_tmp = df_for_split.merge(df_feat, left_on=["Category", "Period"], right_on=["Category_raw", "Period"], how="right")

    # Fix duplicate Sales columns after merge
    if "Sales_x" in df_tmp.columns:
        df_tmp = df_tmp.rename(columns={"Sales_x": "Sales"})
    if "Sales_y" in df_tmp.columns:
        df_tmp = df_tmp.drop(columns=["Sales_y"])

    train, test, test_months = time_aware_split(df_tmp, TEST_FRACTION, MAX_TEST_MONTHS)

    y_col = "Sales"
    helper_cols = ["Sales", "Category_raw", "Category"] + [c for c in ["Period", "_raw_date"] if c in train.columns]
    X_train = train.drop(columns=[c for c in helper_cols if c in train.columns])
    y_train = train[y_col]
    X_test  = test.drop(columns=[c for c in helper_cols if c in test.columns])
    y_test  = test[y_col]

    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test  = X_test.apply(pd.to_numeric, errors="coerce")
    X_train = X_train.dropna(axis=1, how="all")
    X_test  = X_test.dropna(axis=1, how="all")
    mask_train = X_train.notna().all(axis=1)
    mask_test  = X_test.notna().all(axis=1)
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_test,  y_test  = X_test[mask_test],  y_test[mask_test]

    print("Training features:", list(X_train.columns))
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    if len(X_train) == 0 or len(X_test) == 0:
        raise SystemExit("Empty train or test set after feature engineering.")

    model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.9, random_state=RANDOM_SEED, n_jobs=0)
    model.fit(X_train, y_train)

    model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.9, random_state=RANDOM_SEED, n_jobs=0)
    model.fit(X_train, y_train)

    # Save trained model
    model.save_model(MODEL_XGB)
    print(f"[Saved] XGBoost model -> {MODEL_XGB}")
    try:
        joblib.dump(model, MODEL_PKL)
        print(f"[Saved] Pickled model -> {MODEL_PKL}")
    except Exception as e:
        print("[Warn] Could not pickle model:", e)   

    y_pred = model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nOverall RMSE: {rmse:,.3f}")

    test_eval = test.loc[mask_test, ["Category_raw"]].copy()
    test_eval["y_true"] = y_test.values
    test_eval["y_pred"] = y_pred
    per_cat = test_eval.groupby("Category_raw").apply(lambda g: sqrt(mean_squared_error(g["y_true"], g["y_pred"]))).reset_index(name="RMSE").sort_values("RMSE")
    print("\nPer-category RMSE (test window):")
    print(per_cat)

    out_pred = test.loc[mask_test, ["Category_raw"]].copy()
    out_pred["Period"] = test_months[0]
    out_pred["y_true"] = y_test.values
    out_pred["y_pred"] = y_pred
    out_pred.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"[Saved] Test predictions -> {OUTPUT_PREDICTIONS}")

if __name__ == "__main__":
    main()
