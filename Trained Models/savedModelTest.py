
"""
    python savedModelTest.py --model model.pkl --input test_set.csv --output preds.csv
    python savedModelTest.py -m model.xgb -i data.csv         # prints first rows to stdout

"""
from pathlib import Path
import argparse
import sys
import warnings

import numpy as np
import pandas as pd


try:
    import joblib
except Exception:
    joblib = None
import pickle

try:
    import xgboost as xgb
except Exception:
    xgb = None


def load_model(model_path: Path):
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")

    suffix = p.suffix.lower()
    # common joblib/pickle formats
    if suffix in {".pkl", ".pickle", ".joblib"}:
        if joblib:
            try:
                return joblib.load(str(p))
            except Exception:
                pass
        # fallback to pickle
        with open(p, "rb") as f:
            return pickle.load(f)

    # XGBoost native model formats
    if suffix in {".xgb", ".model", ".bin", ".json"}:
        if xgb is None:
            raise RuntimeError("xgboost is not installed but the model file looks like an XGBoost model.")
        booster = xgb.Booster()
        booster.load_model(str(p))
        return booster

    # unknown extension: try joblib/pickle first, then try xgboost Booster if available
    if joblib:
        try:
            return joblib.load(str(p))
        except Exception:
            pass
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass
    if xgb is not None:
        try:
            booster = xgb.Booster()
            booster.load_model(str(p))
            return booster
        except Exception:
            pass

    raise RuntimeError(f"Could not load model from {p}. Unsupported format or missing dependency.")


def align_dataframe_for_model(df: pd.DataFrame, model):
    """
    If model exposes feature names (sklearn's feature_names_in_), subset/align DataFrame.
    Missing columns are filled with zeros. Extra columns are dropped.
    If no feature metadata is available, return df as-is.
    """
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        if missing:
            warnings.warn(f"Missing columns for model: {missing}. Filling with zeros.")
            for c in missing:
                df[c] = 0
        if extra:
            df = df[expected]
        else:
            df = df[expected]
        return df
    # XGBoost sklearn wrappers also sometimes expose feature_names_in_
    if hasattr(model, "get_booster"):  # XGBClassifier/XGBRegressor wrappers
        # try to use underlying booster feature_names if available
        try:
            booster = model.get_booster()
            meta_names = booster.feature_names
            if meta_names:
                expected = list(meta_names)
                missing = [c for c in expected if c not in df.columns]
                if missing:
                    warnings.warn(f"Missing columns for model: {missing}. Filling with zeros.")
                    for c in missing:
                        df[c] = 0
                df = df[expected]
                return df
        except Exception:
            pass
    # No metadata found; return as-is
    return df


def predict_with_model(model, df: pd.DataFrame):
    # XGBoost native Booster
    if xgb is not None and isinstance(model, xgb.Booster):
        # xgboost expects a DMatrix; preserve column names if provided by booster
        dmat = xgb.DMatrix(df.values, feature_names=list(df.columns))
        preds = model.predict(dmat)
        return preds

    # scikit-learn-like models (including XGBClassifier/XGBRegressor wrappers)
    if hasattr(model, "predict"):
        preds = model.predict(df)
        # also include probabilities if available
        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba(df)
            except Exception:
                prob = None
            return preds, prob
        return preds, None

    raise RuntimeError("Loaded model does not have a compatible prediction interface.")


def main(argv):
    p = argparse.ArgumentParser(description="Load a model and predict on CSV input.")
    p.add_argument("-m", "--model", required=True, help="Path to model file (.pkl .joblib .xgb .model ...)")
    p.add_argument("-i", "--input", required=True, help="Path to input CSV. First row should be header with feature names.")
    p.add_argument("-o", "--output", help="Path to output CSV. If omitted prints preview to stdout.")
    p.add_argument("--no-index", action="store_true", help="Do not write index column in output CSV.")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose messages.")
    args = p.parse_args(argv)

    model_path = Path(args.model)
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Input CSV not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    if args.verbose:
        print(f"Loading model from {model_path}...")

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(3)

    if args.verbose:
        print("Reading input CSV...")
    df = pd.read_csv(str(input_path))

    if df.shape[0] == 0:
        print("Input CSV contains no rows.", file=sys.stderr)
        sys.exit(4)

    if args.verbose:
        print(f"Input shape: {df.shape}")

    df_aligned = align_dataframe_for_model(df.copy(), model)

    # If alignment removed columns, make sure shape matches expected
    try:
        preds = predict_with_model(model, df_aligned)
    except Exception as e:
        print(f"Prediction failed: {e}", file=sys.stderr)
        sys.exit(5)

    # Normalize output: preds may be a single array or tuple (preds, prob)
    if isinstance(preds, tuple):
        pred_vals, prob = preds
    else:
        pred_vals, prob = preds, None

    # Build results DataFrame
    results = pd.DataFrame({"prediction": np.asarray(pred_vals).ravel()})
    if prob is not None:
        prob = np.asarray(prob)
        # if multiclass, create prob_0, prob_1, ...
        if prob.ndim == 2 and prob.shape[1] > 1:
            for i in range(prob.shape[1]):
                results[f"prob_{i}"] = prob[:, i]
        else:
            # binary or single-col probabilities
            results["prob"] = prob.ravel()

    # include original index and optionally some original cols (user can change as needed)
    out_df = pd.concat([df.reset_index(drop=True), results], axis=1)

    if args.output:
        out_path = Path(args.output)
        out_df.to_csv(str(out_path), index=not args.no_index)
        if args.verbose:
            print(f"Wrote predictions to {out_path}")
    else:
        # Print a small preview to stdout
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
# ...existing code...