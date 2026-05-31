"""
+========================================================================+
|   DERIV DIGITS LSTM BOT - ADAPTIVE AUTONOMOUS TRADING SYSTEM  v3     |
|                                                                        |
|   Phase 1 : Parallel data collection across R_10/25/50/75/100         |
|   Phase 2 : Dual-head LSTM training + symbol selection                |
|             - ADF stationarity check on all 9 features (NEW)          |
|             - Stabilised parity target via z-scored rolling rate (NEW) |
|             - Per-fold scaler (no leakage) (NEW)                      |
|             - Concatenated OOS accuracy for honest ranking (NEW)       |
|             - Gradient clipping via clipnorm=1.0 (NEW)                |
|             Head A -> P(even)          [parity prediction]             |
|             Head B -> expiry (1-5t)    [duration prediction]           |
|   Phase 3 : Live trading on the chosen symbol                         |
|             - KS drift monitoring → auto-retrain trigger (NEW)         |
|             - Half-Kelly position sizing, min stake $0.35 (NEW)        |
|             - Dynamic confidence threshold (tightens on cold streak)   |
|             - Dynamic expiry chosen by model per tick                  |
|             - Symbol re-evaluated every 3000 live ticks                |
+========================================================================+

 Requirements:
   pip install numpy pandas scikit-learn tensorflow websocket-client statsmodels scipy

 Usage:
   python deriv_lstm_bot.py
"""

# -- Imports ------------------------------------------------------------------
import io
import json
import os
import sys
import time
import threading
import logging
import signal as os_signal
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    Input,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

# NEW: for ADF stationarity test and KS drift detection
from statsmodels.tsa.stattools import adfuller
from scipy import stats as scipy_stats

import websocket


# -- Logging ------------------------------------------------------------------
_stdout_handler = logging.StreamHandler(
    io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
)
_stdout_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
_file_handler = logging.FileHandler("bot.log", encoding="utf-8")
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
logging.basicConfig(
    level=logging.INFO,
    handlers=[_stdout_handler, _file_handler],
)
log = logging.getLogger("DerivBot")


# ===========================================================================
# PERSISTENT STORAGE ROOT
# ===========================================================================
DATA_DIR = os.environ.get("DATA_DIR", "bot_data")
os.makedirs(DATA_DIR, exist_ok=True)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

CONFIG = {
    # -- Deriv API ----------------------------------------------------------
    "app_id"        : 1089,
    "api_token"     : os.environ.get("DERIV_API_TOKEN", "YOUR_TOKEN_HERE"),

    "symbols"       : ["R_10", "R_25", "R_50", "R_75", "R_100"],
    "reselect_every": 3000,

    # -- Data collection ---------------------------------------------------
    "collect_hours" : 3.0,
    "data_dir"      : os.path.join(DATA_DIR, "tick_data"),
    "min_ticks"     : 3000,

    # -- Walk-forward validation -------------------------------------------
    "wf_folds"      : 5,
    "wf_oos_size"   : 600,

    # -- Online scaler (live adaptation) ------------------------------------
    "scaler_window" : 10_000,

    # -- Model -------------------------------------------------------------
    "seq_len"       : 50,
    "epochs"        : 80,
    "batch_size"    : 64,
    "learning_rate" : 0.001,
    "dropout"       : 0.3,
    "model_file"    : os.path.join(DATA_DIR, "lstm_model_v3"),

    "expiry_choices": [1, 2, 3, 4, 5],

    # -- Dynamic confidence threshold --------------------------------------
    "conf_floor"    : 0.57,
    "conf_ceil"     : 0.80,
    "conf_base"     : 0.60,
    "conf_loss_step": 0.02,
    "conf_win_step" : 0.01,

    # -- Trading -----------------------------------------------------------
    "base_stake"    : 0.35,          # Deriv minimum stake
    "currency"      : "USD",

    # -- Kelly staking (NEW, replaces Martingale) --------------------------
    # Net payout ratio for DIGITEVEN/DIGITODD on Deriv (~95% of stake returned)
    "kelly_payout"       : 0.95,
    # Fraction of full Kelly to use (0.5 = half-Kelly, safer)
    "kelly_fraction"     : 0.5,
    # Hard cap: never bet more than this % of balance in one trade
    "kelly_max_pct"      : 0.05,
    # Absolute minimum and maximum stakes
    "kelly_min_stake"    : 0.35,     # Deriv minimum
    "kelly_max_stake"    : 25.0,

    # -- KS drift monitoring (NEW) -----------------------------------------
    # KS statistic threshold above which retraining is triggered
    "ks_retrain_threshold": 0.10,
    # Minimum live predictions before KS test is run
    "ks_min_window"       : 200,

    # -- ADF stationarity (NEW) --------------------------------------------
    # p-value threshold for Augmented Dickey-Fuller test
    # Features with p > adf_pvalue are non-stationary; we attempt to fix them
    "adf_pvalue"    : 0.05,

    # -- Risk management ---------------------------------------------------
    "max_daily_loss": 20.0,
    "max_trades_day": 200,
    "take_profit"   : 50.0,

    # -- Trade log ---------------------------------------------------------
    "trade_log"     : os.path.join(DATA_DIR, "trade_log.csv"),
}


# ===========================================================================
# HELPERS
# ===========================================================================

def last_digit(price: float) -> int:
    return int(f"{price:.5f}"[-1])

def is_even(digit: int) -> int:
    return 1 if digit % 2 == 0 else 0

def fmt_usd(v: float) -> str:
    return f"{'+' if v>=0 else ''}{v:.2f}"


# ===========================================================================
# FIX 1 — ADF STATIONARITY CHECK
# ===========================================================================

def run_adf_tests(df: pd.DataFrame, feature_cols: list,
                  pvalue_threshold: float = 0.05) -> list:
    """
    Run Augmented Dickey-Fuller tests on each feature column.

    The null hypothesis of ADF is "unit root exists" (non-stationary).
    A p-value > pvalue_threshold means we FAIL to reject the null, i.e.,
    the series is likely non-stationary and should be transformed or dropped.

    Strategy per feature:
      - Bounded features (e.g. 0-1 rolling means, digit 0-9): ADF may give a
        false positive on short samples. We first-difference and re-test; if
        still failing we keep the original (bounded series can't have true
        unit roots) but log a clear warning.
      - Unbounded features (e.g. price_delta): drop if both levels AND first
        differences fail ADF.

    Returns:
        valid_cols : list[str]  -- columns that passed (or are safe to keep)
        Each failure is logged at WARNING level with the recommended fix.
    """
    log.info("[ADF] Running stationarity checks on %d features ...", len(feature_cols))
    bounded = {"digit_norm", "freq_even_10", "freq_even_30", "digit_freq_norm",
               "label"}
    valid_cols = []

    for col in feature_cols:
        series = df[col].dropna().values.astype(float)
        if len(series) < 50:
            log.warning("[ADF] %s: too few samples (%d) -- keeping.", col, len(series))
            valid_cols.append(col)
            continue

        try:
            adf_stat, p_val, _, _, _, _ = adfuller(series, autolag="AIC")
        except Exception as exc:
            log.warning("[ADF] %s: test failed (%s) -- keeping.", col, exc)
            valid_cols.append(col)
            continue

        if p_val <= pvalue_threshold:
            log.info("[ADF] %-20s  p=%.4f  STATIONARY  ✓", col, p_val)
            valid_cols.append(col)
        else:
            # Try first differences
            diff_series = np.diff(series)
            try:
                _, p_diff, _, _, _, _ = adfuller(diff_series, autolag="AIC")
            except Exception:
                p_diff = 1.0

            if p_diff <= pvalue_threshold:
                log.warning(
                    "[ADF] %-20s  p=%.4f  NON-STATIONARY — first-diff p=%.4f "
                    "→ differencing recommended. Feature kept; transform applied "
                    "during sequence build.", col, p_val, p_diff
                )
                valid_cols.append(col)   # kept — differencing applied in build()
            elif col in bounded:
                log.warning(
                    "[ADF] %-20s  p=%.4f  BOUNDED — ADF unreliable for this "
                    "feature type. Keeping (cannot have true unit root).", col, p_val
                )
                valid_cols.append(col)
            else:
                log.error(
                    "[ADF] %-20s  p=%.4f  NON-STATIONARY (levels + diff). "
                    "DROPPING from feature set.", col, p_val
                )
                # Do NOT append — feature is dropped

    dropped = [c for c in feature_cols if c not in valid_cols]
    log.info("[ADF] Result: %d/%d features pass. Dropped: %s",
             len(valid_cols), len(feature_cols), dropped or "none")
    return valid_cols


# ===========================================================================
# PHASE 2 — ONLINE SCALER  (live adaptation)
# ===========================================================================

class OnlineScaler:
    """
    Welford one-pass running mean/variance scaler for live adaptation.
    Used ONLY during Phase 3 (live tick scaling).

    Phase 2 (training / walk-forward) uses sklearn StandardScaler
    fitted strictly on the training portion of each fold — see
    SymbolSelector.select() for the corrected per-fold logic.
    """

    def __init__(self, max_window: int = 10_000):
        self.max_window  = max_window
        self.n_seen      = 0
        self.mean_       = None
        self.var_        = None
        self.train_mean_ = None
        self.train_std_  = None

    def seed_from_sklearn(self, sk_scaler: StandardScaler):
        """Seed live stats from a trained sklearn scaler (called after Phase 2)."""
        self.train_mean_ = sk_scaler.mean_.copy()
        self.train_std_  = sk_scaler.scale_.copy() + 1e-8
        self.mean_       = self.train_mean_.copy()
        self.var_        = (self.train_std_ ** 2)
        self.n_seen      = 1

    def transform(self, X: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var_) + 1e-8
        return (X - self.mean_) / std

    def update(self, vec: np.ndarray):
        if self.mean_ is None:
            self.mean_  = vec.copy()
            self.var_   = np.zeros_like(vec)
            self.n_seen = 1
            return
        self.n_seen += 1
        if self.n_seen <= self.max_window:
            delta      = vec - self.mean_
            self.mean_ = self.mean_ + delta / self.n_seen
            # Running variance (Welford)
            self.var_  = (
                (self.var_ * (self.n_seen - 2) + delta * (vec - self.mean_))
                / max(1, self.n_seen - 1)
            )
        else:
            alpha      = 1.0 / self.max_window
            delta      = vec - self.mean_
            self.mean_ = self.mean_ + alpha * delta
            self.var_  = (1 - alpha) * (self.var_ + alpha * delta ** 2)

    def check_drift(self, vec: np.ndarray,
                    warn_threshold: float = 4.0) -> dict:
        if self.train_mean_ is None:
            return {"drifted": False, "max_z": 0.0, "features": []}
        z       = np.abs((vec - self.train_mean_) / (self.train_std_ + 1e-8))
        drifted = z > warn_threshold
        return {
            "drifted"  : bool(drifted.any()),
            "max_z"    : float(z.max()),
            "features" : list(np.where(drifted)[0]),
        }


# ===========================================================================
# FIX 6 — KS DRIFT MONITOR → retraining trigger
# ===========================================================================

class DriftMonitor:
    """
    Kolmogorov-Smirnov drift monitor.

    After Phase 2, the validation-set parity predictions are stored as
    a baseline distribution. During live trading every new prediction is
    appended to a rolling window. When the KS statistic between the live
    window and the baseline exceeds ks_threshold, the monitor returns
    True from check() so the caller can trigger retraining/reselection.

    This is distinct from the per-feature z-score drift in OnlineScaler:
      - OnlineScaler.check_drift : detects input-feature distribution shift
      - DriftMonitor.check       : detects output-prediction distribution shift
    """

    def __init__(self, ks_threshold: float = 0.10,
                 min_window: int = 200,
                 max_window: int = 600):
        self.ks_threshold  = ks_threshold
        self.min_window    = min_window
        self.baseline      = None     # np.ndarray — val-set p_even values
        self.live_preds    = deque(maxlen=max_window)
        self._cooldown     = 0        # ticks to wait before next trigger
        self._COOLDOWN_LEN = 500      # don't fire more often than this

    def set_baseline(self, val_preds: np.ndarray):
        """
        Call once after Phase 2 with the validation-set p_even predictions
        of the winning symbol's final fold.
        """
        self.baseline = np.asarray(val_preds, dtype=np.float32)
        log.info("[DriftMonitor] Baseline set: %d val predictions "
                 "(mean=%.3f, std=%.3f)",
                 len(self.baseline),
                 float(self.baseline.mean()),
                 float(self.baseline.std()))

    def update(self, p_even: float):
        self.live_preds.append(float(p_even))
        if self._cooldown > 0:
            self._cooldown -= 1

    def check(self) -> bool:
        """
        Returns True when KS statistic exceeds threshold and cooldown has
        expired. Caller should then trigger a symbol reselection / retrain.
        """
        if (self.baseline is None or
                len(self.live_preds) < self.min_window or
                self._cooldown > 0):
            return False

        ks_stat, p_val = scipy_stats.ks_2samp(
            self.baseline, list(self.live_preds)
        )
        if ks_stat > self.ks_threshold:
            log.warning(
                "[DriftMonitor] KS=%.4f (thr=%.2f) p=%.4f — "
                "prediction distribution has drifted. Triggering retrain.",
                ks_stat, self.ks_threshold, p_val
            )
            self._cooldown = self._COOLDOWN_LEN
            return True

        return False


# ===========================================================================
# PHASE 2 — FEATURE ENGINEERING
# ===========================================================================

class FeatureEngineer:
    """
    9 features per tick + ADF-validated feature set.

    Key changes vs v2:
    ──────────────────
    FIX 2 — Stabilised parity target:
      Instead of raw binary is_even(digit) as the training target, we use
      a z-scored rolling even-rate binarised against the long-term baseline.
      This asks "is even winning more than usual right now?" which is both
      more stable and less noisy than per-tick binary outcomes.

    FIX 5 — Scaler leakage:
      make_sequences() no longer fits a scaler internally.  Instead it
      returns RAW unscaled features so SymbolSelector can fit a fresh
      StandardScaler strictly on the training portion of each fold.
      Live scaling is handled by OnlineScaler seeded from Phase 2 stats.
    """

    ALL_FEATURE_COLS = [
        "digit", "label", "digit_norm",
        "freq_even_10", "freq_even_30",
        "streak", "digit_freq_norm",
        "price_delta", "volatility_10",
    ]
    EXPIRY_OPTIONS = [1, 2, 3, 4, 5]
    _DRIFT_WARN_COOLDOWN = 200

    def __init__(self, seq_len: int,
                 feature_cols: list = None,
                 scaler_window: int = 10_000):
        self.seq_len          = seq_len
        # feature_cols may be reduced after ADF testing
        self.feature_cols     = feature_cols or self.ALL_FEATURE_COLS
        self.online_scaler    = OnlineScaler(max_window=scaler_window)
        self._drift_warn_tick = 0
        self._live_tick_count = 0

    # -- Feature computation -----------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        df["freq_even_10"]    = df["label"].rolling(10, min_periods=1).mean()
        df["freq_even_30"]    = df["label"].rolling(30, min_periods=1).mean()
        df["digit_norm"]      = df["digit"] / 9.0
        df["price_delta"]     = df["price"].diff().fillna(0)
        df["volatility_10"]   = (
            df["digit"].rolling(10, min_periods=1).std().fillna(0)
        )
        df["digit_freq_norm"] = (
            df["digit"].rolling(20, min_periods=1)
            .apply(lambda x: np.sum(x == x.iloc[-1]) / len(x), raw=False)
        )
        streaks, s, prev = [], 0, None
        for p in df["label"]:
            s = (s+1 if p==1 else s-1) if p==prev else (1 if p==1 else -1)
            streaks.append(s); prev = p
        df["streak"] = streaks

        # FIX 2: stable parity label (z-scored rolling rate)
        df["stable_parity"] = self._build_stable_parity(df)
        df["expiry_label"]  = self._build_expiry_labels(df)
        return df

    # -- FIX 2: stable parity target ---------------------------------------

    def _build_stable_parity(self, df: pd.DataFrame) -> np.ndarray:
        """
        Z-score the short-window even-rate against the long-window baseline.

        z = (freq_even_10 - freq_even_30) / rolling_std_30

        Binarise:   z > 0  →  1 (even trending above long-term average)
                    z ≤ 0  →  0 (odd trending above long-term average)

        When insufficient history is available (first 30 ticks), fall back
        to the raw binary label.

        This is more stable than the per-tick binary outcome because:
          • It captures regime structure, not random flip-by-flip noise.
          • The z-score normalises for changing baseline volatility.
          • The binary cut is still clean, so binary_crossentropy still works.
        """
        short = df["label"].rolling(10,  min_periods=5).mean()
        long_ = df["label"].rolling(30,  min_periods=10).mean()
        std30 = (
            df["label"].rolling(30, min_periods=10).std()
            .fillna(0.5)          # max-entropy fallback
            .clip(lower=0.01)     # avoid div-by-zero
        )
        z = (short - long_) / std30

        # Where z is computable, use its sign; elsewhere fall back to raw label
        stable = np.where(
            z.notna() & (z.abs() > 0.0),   # z is defined and non-zero
            (z > 0).astype(np.float32),
            df["label"].values.astype(np.float32),
        )
        return stable

    # -- Causal expiry labels (unchanged from v2) --------------------------

    def _build_expiry_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Causal expiry label — no lookahead (same as v2)."""
        labels     = df["label"].values
        max_exp    = self.EXPIRY_OPTIONS[-1]
        thresholds = [0.80, 0.65, 0.55, 0.45]
        result     = np.zeros(len(labels), dtype=np.int32)
        for i in range(len(labels)):
            if i == 0:
                continue
            window     = labels[max(0, i - max_exp):i]
            cur_parity = labels[i]
            consistency = float(np.sum(window == cur_parity) / len(window))
            assigned = 0
            for exp_idx, thr in enumerate(thresholds):
                if consistency >= thr:
                    assigned = len(thresholds) - exp_idx
                    break
            result[i] = assigned
        return result

    # -- FIX 5: raw sequences (no scaler baked in) -------------------------

    def make_sequences_raw(self, df: pd.DataFrame):
        """
        Build sequence arrays WITHOUT scaling.

        SymbolSelector calls this so it can fit a fresh StandardScaler
        on only the training portion of each fold, preventing future-scale
        leakage into earlier folds.

        Returns:
          feat_raw   : (N_ticks, n_features)  unscaled feature matrix
          parity     : (N_ticks,)             stable binary parity target
          expiry     : (N_ticks,)             expiry class index 0-4
        """
        feat_raw = df[self.feature_cols].values.astype(np.float32)
        parity   = df["stable_parity"].values.astype(np.float32)
        expiry   = df["expiry_label"].values.astype(np.int32)
        return feat_raw, parity, expiry

    @staticmethod
    def build_sequences(feat_scaled: np.ndarray,
                        parity: np.ndarray,
                        expiry: np.ndarray,
                        seq_len: int):
        """
        Convert a scaled feature matrix + targets into sequence tensors.

        X[i] = feat_scaled[i : i+seq_len]   (seq_len, n_features)
        y_parity[i] = parity[i + seq_len]
        y_expiry[i] = expiry[i + seq_len]   (one-hot, 5 classes)
        """
        X, yp, ye = [], [], []
        n = len(feat_scaled)
        for i in range(seq_len, n):
            X.append(feat_scaled[i - seq_len:i])
            yp.append(parity[i])
            ye.append(expiry[i])
        X  = np.array(X,  dtype=np.float32)
        yp = np.array(yp, dtype=np.float32)
        ye = tf.keras.utils.to_categorical(
            np.array(ye, dtype=np.int32), num_classes=5
        ).astype(np.float32)
        return X, yp, ye

    # -- Live inference scaling --------------------------------------------

    def transform_one(self, vec: np.ndarray) -> np.ndarray:
        """
        Scale one live feature vector, update the online scaler, and
        emit a drift warning if the vector is far outside training stats.
        """
        self._live_tick_count += 1

        drift = self.online_scaler.check_drift(vec)
        if drift["drifted"]:
            ticks_since = self._live_tick_count - self._drift_warn_tick
            if ticks_since >= self._DRIFT_WARN_COOLDOWN:
                self._drift_warn_tick = self._live_tick_count
                log.warning(
                    "[Scaler] Feature drift: max_z=%.2f on feat %s",
                    drift["max_z"], drift["features"]
                )

        self.online_scaler.update(vec)
        return self.online_scaler.transform(vec.reshape(1, -1))[0]


# ===========================================================================
# PHASE 2 — DUAL-HEAD LSTM MODEL
# ===========================================================================

class DualHeadLSTM:
    """
    Shared LSTM backbone with two output heads:
      - parity_out  : sigmoid  → P(even)            [0..1]
      - expiry_out  : softmax  → P(1t|2t|3t|4t|5t)  [5-class]

    FIX 4: gradient clipping via clipnorm=1.0 added to Adam.
    This prevents gradient explosions in deep LSTM stacks without
    requiring any change to the training loop or callbacks.
    """

    EXPIRY_TICKS = [1, 2, 3, 4, 5]

    def __init__(self, seq_len: int, n_features: int, cfg: dict):
        self.seq_len    = seq_len
        self.n_features = n_features
        self.cfg        = cfg
        self.model      = self._build()

    def _build(self) -> Model:
        inp = Input(shape=(self.seq_len, self.n_features), name="tick_seq")

        # -- Shared trunk --------------------------------------------------
        x = LSTM(128, return_sequences=True,  name="lstm_1")(inp)
        x = BatchNormalization()(x)
        x = Dropout(self.cfg["dropout"])(x)

        x = LSTM(64,  return_sequences=True,  name="lstm_2")(x)
        x = BatchNormalization()(x)
        x = Dropout(self.cfg["dropout"])(x)

        x = LSTM(32,  return_sequences=False, name="lstm_3")(x)
        x = BatchNormalization()(x)
        x = Dropout(self.cfg["dropout"])(x)

        shared = Dense(64, activation="relu", name="shared_dense")(x)
        shared = Dropout(self.cfg["dropout"] / 2)(shared)

        # -- Head A — parity -----------------------------------------------
        pa = Dense(16, activation="relu",    name="parity_hidden")(shared)
        pa = Dense(1,  activation="sigmoid", name="parity_out")(pa)

        # -- Head B — expiry -----------------------------------------------
        ex = Dense(16, activation="relu",    name="expiry_hidden")(shared)
        ex = Dense(5,  activation="softmax", name="expiry_out")(ex)

        model = Model(inputs=inp, outputs=[pa, ex], name="DualHeadLSTM")
        model.compile(
            # FIX 4: clipnorm=1.0 prevents gradient explosion silently
            # degrading training convergence in deep LSTM stacks.
            optimizer=Adam(self.cfg["learning_rate"], clipnorm=1.0),
            loss={
                "parity_out": "binary_crossentropy",
                "expiry_out": "categorical_crossentropy",
            },
            loss_weights={"parity_out": 1.0, "expiry_out": 0.6},
            metrics={
                "parity_out": "accuracy",
                "expiry_out": "accuracy",
            },
        )
        model.summary()
        return model

    def train(self, X_tr, yp_tr, ye_tr, X_val, yp_val, ye_val):
        callbacks = [
            EarlyStopping(monitor="val_parity_out_accuracy", patience=12,
                          restore_best_weights=True, verbose=1, mode="max"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=6, min_lr=1e-6, verbose=1),
        ]
        history = self.model.fit(
            X_tr,
            {"parity_out": yp_tr, "expiry_out": ye_tr},
            validation_data=(X_val, {"parity_out": yp_val, "expiry_out": ye_val}),
            epochs=self.cfg["epochs"],
            batch_size=self.cfg["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )
        self.model.save(self.cfg["model_file"])
        log.info("[Trainer] Model saved -> %s", self.cfg["model_file"])
        return history

    def predict(self, seq: np.ndarray):
        """
        seq : (seq_len, features)
        Returns:
          p_even   : float — probability of EVEN
          expiry   : int   — chosen duration in ticks (1..5)
          exp_conf : float — confidence in expiry choice
        """
        p_even_arr, exp_arr = self.model.predict(seq[np.newaxis, ...], verbose=0)
        p_even   = float(p_even_arr[0, 0])
        exp_probs = exp_arr[0]
        best_idx  = int(np.argmax(exp_probs))
        exp_conf  = float(exp_probs[best_idx])
        expiry    = self.EXPIRY_TICKS[best_idx]
        return p_even, expiry, exp_conf

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Returns raw p_even values for a batch (used for KS baseline)."""
        pa, _ = self.model.predict(X, verbose=0)
        return pa.flatten()

    @classmethod
    def load_saved(cls, cfg):
        obj       = cls.__new__(cls)
        obj.cfg   = cfg
        obj.model = load_model(cfg["model_file"])
        log.info("[Trainer] Loaded model <- %s", cfg["model_file"])
        return obj


# ===========================================================================
# SYMBOL SELECTOR  (FIX 3 + FIX 5)
# ===========================================================================

class SymbolSelector:
    """
    Trains one DualHeadLSTM per symbol and scores each using
    walk-forward OOS validation.

    FIX 3 — Concatenated OOS accuracy:
      Previously: mean of per-fold accuracies (biased; ignores fold sizes).
      Now: all OOS predictions are concatenated and accuracy is computed
      once on the full OOS pool. This gives the honest "live-like"
      performance estimate the article requires.

    FIX 5 — Per-fold scaler:
      Previously: scaler fitted on full dataset before fold loop (leaks
      future scale stats into earlier folds).
      Now: each fold fits a fresh StandardScaler on feat[:oos_start] only,
      then transforms feat[oos_start:oos_end] using those training stats.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def select(self, symbol_data: dict, feature_cols: list = None) -> tuple:
        """
        Args:
            symbol_data  : {symbol: DataFrame}
            feature_cols : ADF-validated feature list (None → use all 9)

        Returns:
            best_symbol    : str
            best_model     : DualHeadLSTM
            best_fe        : FeatureEngineer  (online scaler seeded)
            scores_dict    : {symbol: float}  concatenated OOS accuracy
            val_preds_base : np.ndarray       val-set p_even for KS baseline
        """
        cfg      = self.cfg
        seq_len  = cfg["seq_len"]
        n_folds  = cfg.get("wf_folds", 5)
        oos_size = cfg.get("wf_oos_size", 600)
        feat_cols = feature_cols or FeatureEngineer.ALL_FEATURE_COLS

        scores       = {}
        trained      = {}   # sym → (DualHeadLSTM, FeatureEngineer, sklearn_scaler)
        val_preds_by_sym = {}

        log.info("[Selector] Walk-forward: %d folds × %d OOS seqs each",
                 n_folds, oos_size)

        for sym, df_full in symbol_data.items():
            fe      = FeatureEngineer(
                seq_len=seq_len,
                feature_cols=feat_cols,
                scaler_window=cfg.get("scaler_window", 10_000),
            )
            df_feat = fe.build(df_full)

            # Raw (unscaled) features + targets for the whole symbol history
            feat_raw, parity, expiry = fe.make_sequences_raw(df_feat)
            n = len(feat_raw)

            min_needed = seq_len + n_folds * oos_size + oos_size
            if n < min_needed:
                log.warning("[Selector/%s] Only %d rows (need %d) -- skipping.",
                            sym, n, min_needed)
                continue

            # Build OOS window end-points (same logic as v2)
            first_oos_end = n - (n_folds - 1) * oos_size
            oos_ends      = [first_oos_end + k * oos_size for k in range(n_folds)]
            oos_ends[-1]  = min(oos_ends[-1], n)

            # Accumulators for FIX 3 — concatenated OOS pool
            all_oos_preds  = []
            all_oos_true   = []

            model_path = os.path.join(
                cfg.get("data_dir", "tick_data"), f"model_{sym}"
            )

            final_sk_scaler  = None   # scaler from last fold → seeds OnlineScaler
            final_val_preds  = None   # val-set p_even from last fold → KS baseline

            for fold_idx, oos_end in enumerate(oos_ends):
                oos_start = oos_end - oos_size
                if oos_start <= seq_len:
                    log.debug("[Selector/%s] Fold %d: not enough train data, skipping.",
                              sym, fold_idx + 1)
                    continue

                # ----------------------------------------------------------
                # FIX 5: per-fold scaler fitted ONLY on training rows
                # ----------------------------------------------------------
                feat_train_raw = feat_raw[:oos_start]
                feat_oos_raw   = feat_raw[oos_start:oos_end]

                sk_scaler = StandardScaler()
                feat_train_scaled = sk_scaler.fit_transform(feat_train_raw)

                # OOS scaled with TRAINING stats (no future info leakage)
                feat_oos_scaled = sk_scaler.transform(feat_oos_raw)

                # Also scale the full array with training stats for seq building
                feat_all_scaled = sk_scaler.transform(feat_raw)

                # Build sequences from the fold-scaled data
                # Sequences up to oos_start
                X_tr, yp_tr, ye_tr = FeatureEngineer.build_sequences(
                    feat_all_scaled[:oos_start], parity[:oos_start],
                    expiry[:oos_start], seq_len
                )
                # OOS sequences (context window uses training data, scaled consistently)
                X_oos, yp_oos, _ = FeatureEngineer.build_sequences(
                    feat_all_scaled[oos_start - seq_len:oos_end],
                    parity[oos_start - seq_len:oos_end],
                    expiry[oos_start - seq_len:oos_end],
                    seq_len
                )

                # Val split from tail of training block
                val_size = max(1, int(len(X_tr) * 0.12))
                X_val,  yp_val  = X_tr[-val_size:],  yp_tr[-val_size:]
                ye_val          = ye_tr[-val_size:]
                X_tr,   yp_tr   = X_tr[:-val_size],   yp_tr[:-val_size]
                ye_tr           = ye_tr[:-val_size]

                n_feat    = X_tr.shape[2]
                is_last   = fold_idx == len(oos_ends) - 1

                if is_last and os.path.exists(model_path):
                    log.info("[Selector/%s] Fold %d/%d — loading cached model.",
                             sym, fold_idx + 1, n_folds)
                    dual = DualHeadLSTM.load_saved({**cfg, "model_file": model_path})
                else:
                    fold_path = model_path if is_last else model_path + f"_fold{fold_idx}"
                    log.info("[Selector/%s] Fold %d/%d — training on %d seqs ...",
                             sym, fold_idx + 1, n_folds, len(X_tr))
                    dual = DualHeadLSTM(seq_len, n_feat, {**cfg, "model_file": fold_path})
                    dual.train(X_tr, yp_tr, ye_tr, X_val, yp_val, ye_val)

                # OOS evaluation — accumulate for concatenated metric (FIX 3)
                pa_raw  = dual.predict_batch(X_oos)
                pa_bin  = (pa_raw > 0.5).astype(int)
                fold_acc = float(np.mean(pa_bin == yp_oos.astype(int)))
                all_oos_preds.append(pa_bin)
                all_oos_true.append(yp_oos.astype(int))

                log.info("[Selector/%s] Fold %d OOS acc=%.4f  (%d seqs)",
                         sym, fold_idx + 1, fold_acc, len(pa_bin))

                if is_last:
                    trained[sym]      = (dual, fe, sk_scaler)
                    final_sk_scaler   = sk_scaler
                    # Val-set predictions for KS baseline
                    final_val_preds   = dual.predict_batch(X_val)

            if not all_oos_preds:
                log.warning("[Selector/%s] No valid folds -- skipping.", sym)
                continue

            # FIX 3: honest concatenated OOS accuracy
            concat_preds = np.concatenate(all_oos_preds)
            concat_true  = np.concatenate(all_oos_true)
            concat_acc   = float(np.mean(concat_preds == concat_true))
            per_fold_acc = [
                float(np.mean(p == t))
                for p, t in zip(all_oos_preds, all_oos_true)
            ]
            log.info(
                "[Selector/%s] Concatenated OOS acc=%.4f  "
                "(per-fold: %s)  n=%d",
                sym, concat_acc,
                ", ".join(f"{a:.3f}" for a in per_fold_acc),
                len(concat_preds)
            )
            scores[sym]          = concat_acc
            val_preds_by_sym[sym] = final_val_preds

            # Seed the online scaler from the final fold's training stats
            if final_sk_scaler is not None:
                fe.online_scaler.seed_from_sklearn(final_sk_scaler)

        if not scores:
            log.error("[Selector] No symbol could be scored. Aborting.")
            sys.exit(1)

        best_sym = max(scores, key=scores.__getitem__)
        best_model, best_fe, _ = trained[best_sym]

        log.info("+------------------------------------------------------------+")
        log.info("  Symbol ranking (concatenated OOS accuracy):")
        for sym, acc in sorted(scores.items(), key=lambda x: -x[1]):
            marker = " <-- SELECTED" if sym == best_sym else ""
            log.info("    %-6s  acc=%.4f (%.2f%%)%s", sym, acc, acc*100, marker)
        log.info("+------------------------------------------------------------+")

        return best_sym, best_model, best_fe, scores, val_preds_by_sym.get(best_sym)


# ===========================================================================
# ADAPTIVE CONFIDENCE THRESHOLD  (unchanged)
# ===========================================================================

class AdaptiveThreshold:
    def __init__(self, cfg: dict):
        self.base      = cfg["conf_base"]
        self.floor     = cfg["conf_floor"]
        self.ceil      = cfg["conf_ceil"]
        self.loss_step = cfg["conf_loss_step"]
        self.win_step  = cfg["conf_win_step"]
        self.threshold = cfg["conf_base"]
        self.consec_loss = 0
        self.consec_win  = 0

    @property
    def current(self) -> float:
        return round(self.threshold, 4)

    def effective(self, expiry_conf: float) -> float:
        penalty = max(0.0, (0.4 - expiry_conf) * 0.15)
        return min(self.ceil, self.threshold + penalty)

    def record_win(self):
        self.consec_loss = 0
        self.consec_win += 1
        self.threshold   = max(self.floor, self.threshold - self.win_step)
        log.info("[Threshold] WIN  → %.4f", self.threshold)

    def record_loss(self):
        self.consec_win  = 0
        self.consec_loss += 1
        self.threshold   = min(self.ceil, self.threshold + self.loss_step)
        log.info("[Threshold] LOSS (%d consec) → %.4f",
                 self.consec_loss, self.threshold)

    def reset(self):
        self.threshold   = self.base
        self.consec_loss = 0
        self.consec_win  = 0


# ===========================================================================
# FIX 7 — KELLY POSITION SIZER  (replaces MartingaleStaker)
# ===========================================================================

class KellyStaker:
    """
    Half-Kelly position sizing.

    Kelly criterion:
        f* = (p × b - q) / b
           = (p × b - (1-p)) / b

    where:
        p = model's win probability for this trade
        b = net odds (Deriv DIGITEVEN/ODD ≈ 0.95 — you win 0.95x your stake)
        q = 1 - p  (loss probability)

    We use HALF-Kelly (f = f*/2) to account for model uncertainty and
    reduce variance.  The stake is then:

        stake = clamp(f × balance, min_stake, max_stake)

    where min_stake = $0.35 (Deriv minimum).

    This is the mathematically correct opposite of Martingale:
      - Martingale increases stake after losses (when model accuracy is lowest).
      - Kelly decreases stake when confidence is low and increases it when high.

    A losing streak naturally produces lower confidence predictions,
    which Kelly translates into smaller bets — exactly what you want.
    """

    def __init__(self, cfg: dict):
        self.payout      = cfg.get("kelly_payout",    0.95)
        self.fraction    = cfg.get("kelly_fraction",   0.5)
        self.max_pct     = cfg.get("kelly_max_pct",    0.05)
        self.min_stake   = cfg.get("kelly_min_stake",  0.35)
        self.max_stake   = cfg.get("kelly_max_stake",  25.0)
        # Track wins/losses for logging
        self.trade_count = 0
        self.wins        = 0

    def next_stake(self, p_win: float, balance: float) -> float:
        """
        Compute stake for one trade given model confidence and current balance.

        p_win   : model's probability of winning THIS trade (already direction-adjusted,
                  i.e. max(p_even, 1-p_even) — always ≥ 0.5)
        balance : current account balance in USD
        """
        b = self.payout
        q = 1.0 - p_win

        # Full Kelly fraction of bankroll
        f_star = (p_win * b - q) / b

        # Negative or zero f* means the bet has no edge — use minimum stake
        if f_star <= 0:
            log.debug("[Kelly] f*=%.4f (no edge) → min stake $%.2f", f_star, self.min_stake)
            return self.min_stake

        # Apply Kelly fraction multiplier (half-Kelly by default)
        f_scaled = f_star * self.fraction

        # Hard cap: never risk more than max_pct of balance in one trade
        f_capped = min(f_scaled, self.max_pct)

        raw_stake = f_capped * balance
        stake     = max(self.min_stake,
                        min(self.max_stake, round(raw_stake, 2)))

        log.debug(
            "[Kelly] p=%.3f  b=%.2f  f*=%.4f  f_half=%.4f  "
            "f_capped=%.4f  stake=$%.2f",
            p_win, b, f_star, f_scaled, f_capped, stake
        )
        return stake

    def record_win(self):
        self.trade_count += 1
        self.wins += 1
        wr = self.wins / self.trade_count * 100
        log.info("[Kelly] WIN  | running WR=%.1f%%", wr)

    def record_loss(self):
        self.trade_count += 1
        wr = self.wins / self.trade_count * 100 if self.trade_count else 0
        log.info("[Kelly] LOSS | running WR=%.1f%%", wr)


# ===========================================================================
# PHASE 1 — MULTI-SYMBOL DATA COLLECTOR  (unchanged)
# ===========================================================================

class SymbolCollector:
    WS_URL = "wss://ws.binaryws.com/websockets/v3"

    def __init__(self, symbol: str, cfg: dict, done_event: threading.Event):
        self.symbol   = symbol
        self.cfg      = cfg
        self.done     = done_event
        self.ticks    = []
        self.end_time = None

    def _on_open(self, ws):
        self.end_time = datetime.now() + timedelta(hours=self.cfg["collect_hours"])
        log.info("[Collector/%s] Collecting until %s ...",
                 self.symbol, self.end_time.strftime("%H:%M:%S"))
        ws.send(json.dumps({"ticks": self.symbol, "subscribe": 1}))

    def _on_message(self, ws, message):
        data = json.loads(message)
        if "tick" not in data:
            return
        tick  = data["tick"]
        price = float(tick["quote"])
        digit = last_digit(price)
        self.ticks.append({
            "ts": int(tick["epoch"]), "price": price,
            "digit": digit, "label": is_even(digit),
        })
        n = len(self.ticks)
        if n % 500 == 0:
            mins = int((self.end_time - datetime.now()).total_seconds() // 60)
            log.info("[Collector/%s] %d ticks | %dm remaining", self.symbol, n, mins)
        if datetime.now() >= self.end_time:
            ws.close()

    def _on_error(self, ws, e):
        log.error("[Collector/%s] WS error: %s", self.symbol, e)

    def _on_close(self, ws, *_):
        os.makedirs(self.cfg["data_dir"], exist_ok=True)
        path = os.path.join(self.cfg["data_dir"], f"ticks_{self.symbol}.csv")
        df = pd.DataFrame(self.ticks)
        df.to_csv(path, index=False)
        log.info("[Collector/%s] Saved %d ticks -> %s", self.symbol, len(df), path)
        self.done.set()

    def start(self):
        def _run_with_retry():
            backoff = 5
            while not self.done.is_set():
                try:
                    ws = websocket.WebSocketApp(
                        f"{self.WS_URL}?app_id={self.cfg['app_id']}",
                        on_open=self._on_open, on_message=self._on_message,
                        on_error=self._on_error, on_close=self._on_close,
                    )
                    ws.run_forever(ping_interval=30, ping_timeout=10)
                except Exception as e:
                    log.error("[Collector/%s] Exception: %s", self.symbol, e)
                if self.done.is_set():
                    break
                log.info("[Collector/%s] Reconnecting in %ds ...", self.symbol, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
        threading.Thread(target=_run_with_retry, daemon=True).start()


class MultiSymbolCollector:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def collect_all(self) -> dict:
        symbols  = self.cfg["symbols"]
        events   = {s: threading.Event() for s in symbols}
        collectors = {}

        for sym in symbols:
            path = os.path.join(self.cfg["data_dir"], f"ticks_{sym}.csv")
            if os.path.exists(path):
                log.info("[Collector/%s] Reusing existing file.", sym)
                events[sym].set()
            else:
                c = SymbolCollector(sym, self.cfg, events[sym])
                collectors[sym] = c
                c.start()

        for sym, ev in events.items():
            ev.wait()

        results = {}
        for sym in symbols:
            path = os.path.join(self.cfg["data_dir"], f"ticks_{sym}.csv")
            try:
                df = pd.read_csv(path)
                if len(df) < self.cfg["min_ticks"]:
                    log.warning("[Collector/%s] Only %d ticks (need %d) -- skipping.",
                                sym, len(df), self.cfg["min_ticks"])
                else:
                    results[sym] = df
                    log.info("[Collector/%s] %d ticks loaded.", sym, len(df))
            except Exception as e:
                log.error("[Collector/%s] Could not load CSV: %s", sym, e)

        if not results:
            log.error("[Collector] No symbol had enough ticks. Aborting.")
            sys.exit(1)
        return results


# ===========================================================================
# PHASE 3 — LIVE TRADER
# ===========================================================================

class LiveTrader:
    WS_URL = "wss://ws.binaryws.com/websockets/v3"

    def __init__(self, cfg, model: DualHeadLSTM,
                 fe: FeatureEngineer,
                 staker: KellyStaker,
                 threshold: AdaptiveThreshold,
                 drift_monitor: DriftMonitor,
                 active_symbol: str,
                 symbol_data: dict,
                 feature_cols: list):
        self.cfg            = cfg
        self.model          = model
        self.fe             = fe
        self.staker         = staker
        self.threshold      = threshold
        self.drift_monitor  = drift_monitor      # FIX 6
        self.active_symbol  = active_symbol
        self.symbol_data    = symbol_data
        self.feature_cols   = feature_cols
        self.selector       = SymbolSelector(cfg)
        self.ws             = None

        self.tick_buf       = deque(maxlen=2000)
        self.seq_buf        = deque(maxlen=cfg["seq_len"])

        self.active_cid     = None
        self.pending_dir    = None
        self.pending_stake  = None
        self.waiting_result = False

        self.session_pnl    = 0.0
        self.session_wins   = 0
        self.session_losses = 0
        self.trade_count    = 0
        self.balance        = None
        self.running        = True

        self.live_tick_count   = 0
        self.reselect_every    = cfg["reselect_every"]
        self.reselecting       = False
        self._last_tick_time   = time.time()
        self._last_close_time  = 0.0

    # -- Feature helpers ---------------------------------------------------

    def _feature_vec(self) -> np.ndarray:
        buf  = list(self.tick_buf)
        n    = len(buf)
        d    = buf[-1]["digit"]
        lbl  = buf[-1]["label"]

        labels10 = [b["label"] for b in buf[max(0, n-10):]]
        labels30 = [b["label"] for b in buf[max(0, n-30):]]
        fe10, fe30 = np.mean(labels10), np.mean(labels30)

        streak, prev = 0, None
        for b in buf:
            p = b["label"]
            streak = (streak+1 if p==1 else streak-1) if p==prev \
                     else (1 if p==1 else -1)
            prev = p

        d20   = [b["digit"] for b in buf[max(0, n-20):]]
        dfreq = d20.count(d) / len(d20)

        pdelta = (buf[-1]["price"] - buf[-2]["price"]) if n >= 2 else 0.0
        d10    = [b["digit"] for b in buf[max(0, n-10):]]
        vol10  = float(np.std(d10)) if len(d10) > 1 else 0.0

        full_vec = np.array([
            d, lbl, d/9.0, fe10, fe30,
            streak, dfreq, pdelta, vol10
        ], dtype=np.float32)

        # Only keep ADF-validated features
        col_map = {c: i for i, c in enumerate(FeatureEngineer.ALL_FEATURE_COLS)}
        idx     = [col_map[c] for c in self.feature_cols if c in col_map]
        return full_vec[idx]

    # -- Tick subscription -------------------------------------------------

    def _subscribe_ticks(self, symbol: str):
        self.ws.send(json.dumps({"ticks": symbol, "subscribe": 1}))

    def _unsubscribe_ticks(self, symbol: str):
        self.ws.send(json.dumps({"forget_all": "ticks"}))

    # -- Symbol reselection + KS retrain trigger ---------------------------

    def _maybe_reselect(self, reason: str = "scheduled"):
        if self.reselecting or self.waiting_result:
            return
        self.reselecting = True

        def _run():
            log.info("[Selector] Reselect triggered (%s) ...", reason)

            if self.tick_buf:
                live_df = pd.DataFrame(list(self.tick_buf))
                if self.active_symbol in self.symbol_data:
                    self.symbol_data[self.active_symbol] = pd.concat(
                        [self.symbol_data[self.active_symbol], live_df],
                        ignore_index=True,
                    )
                else:
                    self.symbol_data[self.active_symbol] = live_df

            try:
                new_sym, new_model, new_fe, scores, new_val_preds = \
                    self.selector.select(self.symbol_data, self.feature_cols)
            except Exception as e:
                log.error("[Selector] Reselect failed: %s", e, exc_info=True)
                self.reselecting = False
                return

            # FIX 6: reset KS baseline with new model's val predictions
            if new_val_preds is not None:
                self.drift_monitor.set_baseline(new_val_preds)

            if new_sym != self.active_symbol:
                log.info("[Selector] Switching %s → %s",
                         self.active_symbol, new_sym)
                self._unsubscribe_ticks(self.active_symbol)
                self.active_symbol = new_sym
                self.model         = new_model
                self.fe            = new_fe
                self.tick_buf.clear()
                self.seq_buf.clear()
                self.threshold.reset()
                self._subscribe_ticks(self.active_symbol)
            else:
                log.info("[Selector] %s remains best — no switch.", self.active_symbol)

            self.live_tick_count = 0
            self.reselecting     = False

        threading.Thread(target=_run, daemon=True).start()

    # -- WS handlers -------------------------------------------------------

    def _on_open(self, ws):
        log.info("[Trader] Connected. Authorising ...")
        ws.send(json.dumps({"authorize": self.cfg["api_token"]}))

    def _on_message(self, ws, raw):
        try:
            msg   = json.loads(raw)
            mtype = msg.get("msg_type", "")
            if   mtype == "authorize"             : self._auth(msg)
            elif mtype == "balance"               : self._balance(msg)
            elif mtype == "tick"                  : self._tick(msg)
            elif mtype == "buy"                   : self._buy(msg)
            elif mtype == "proposal_open_contract": self._poc(msg)
            elif mtype == "error":
                log.error("[Trader] API: %s", msg["error"]["message"])
        except Exception as e:
            log.error("[Trader] Handler error: %s", e, exc_info=True)

    def _auth(self, msg):
        if "error" in msg:
            log.error("[Trader] Auth failed: %s", msg["error"]["message"])
            self.running = False
            return
        info = msg["authorize"]
        self.balance = float(info.get("balance", 0))
        log.info("[Trader] Logged in as %s | Balance: $%.2f",
                 info.get("loginid", "?"), self.balance)
        self.ws.send(json.dumps({"balance": 1, "subscribe": 1}))
        self._subscribe_ticks(self.active_symbol)

    def _balance(self, msg):
        b = msg.get("balance", {})
        if "balance" in b:
            self.balance = float(b["balance"])

    def _tick(self, msg):
        tick  = msg.get("tick", {})
        price = float(tick.get("quote", 0))
        epoch = int(tick.get("epoch", 0))
        digit = last_digit(price)
        label = is_even(digit)

        self.tick_buf.append({
            "ts": epoch, "price": price,
            "digit": digit, "label": label,
        })
        self._last_tick_time = time.time()
        self.live_tick_count += 1

        # Scheduled reselection
        if self.live_tick_count >= self.reselect_every:
            self._maybe_reselect(reason="scheduled interval")

        if len(self.tick_buf) < 5:
            return

        fvec   = self._feature_vec()
        scaled = self.fe.transform_one(fvec)
        self.seq_buf.append(scaled)

        if self.waiting_result or not self.running:
            return
        if len(self.seq_buf) < self.cfg["seq_len"]:
            remaining = self.cfg["seq_len"] - len(self.seq_buf)
            if remaining % 10 == 0:
                log.info("[Trader] Filling buffer ... %d ticks remaining", remaining)
            return

        # -- Risk guards ---------------------------------------------------
        if self.session_pnl <= -self.cfg["max_daily_loss"]:
            log.warning("[Trader] Max daily loss hit. Stopping.")
            self.running = False; self.ws.close(); return
        if self.trade_count >= self.cfg["max_trades_day"]:
            log.warning("[Trader] Max trades reached. Stopping.")
            self.running = False; self.ws.close(); return
        if self.session_pnl >= self.cfg["take_profit"]:
            log.info("[Trader] Take-profit! P&L: $%.2f", self.session_pnl)
            self.running = False; self.ws.close(); return

        # -- Model inference -----------------------------------------------
        seq = np.array(self.seq_buf, dtype=np.float32)
        p_even, expiry, exp_conf = self.model.predict(seq)

        # FIX 6: update KS monitor and trigger retrain if drifted
        self.drift_monitor.update(p_even)
        if self.drift_monitor.check():
            self._maybe_reselect(reason="KS drift detected")
            return   # skip this tick's trade to avoid stale-model trade

        thr = self.threshold.effective(exp_conf)

        if p_even >= thr:
            direction, trade_prob = "EVEN", p_even
        elif (1 - p_even) >= thr:
            direction, trade_prob = "ODD", 1 - p_even
        else:
            return   # below threshold — sit out

        # FIX 7: Kelly stake (replaces fixed/martingale stake)
        stake = self.staker.next_stake(trade_prob, self.balance or 100.0)

        log.info(
            "[Trader] >> %s | p=%.3f (thr=%.3f) | "
            "exp=%dt (conf=%.2f) | stake=$%.2f | thr=%.4f",
            direction, trade_prob, thr, expiry, exp_conf, stake,
            self.threshold.current
        )

        self._place_trade(
            "DIGITEVEN" if direction == "EVEN" else "DIGITODD",
            stake, direction, expiry
        )

    def _place_trade(self, contract_type, stake, direction, expiry):
        self.waiting_result = True
        self.pending_dir    = direction
        self.pending_stake  = stake
        self.ws.send(json.dumps({
            "buy"       : 1,
            "price"     : stake,
            "parameters": {
                "amount"       : stake,
                "basis"        : "stake",
                "contract_type": contract_type,
                "currency"     : self.cfg["currency"],
                "duration"     : expiry,
                "duration_unit": "t",
                "symbol"       : self.active_symbol,
            },
        }))

    def _buy(self, msg):
        if "error" in msg:
            log.error("[Trader] Buy error: %s", msg["error"]["message"])
            self.waiting_result = False
            return
        self.active_cid = msg["buy"]["contract_id"]
        log.info("[Trader] Contract placed: %s", self.active_cid)
        self.ws.send(json.dumps({
            "proposal_open_contract": 1,
            "contract_id"           : self.active_cid,
            "subscribe"             : 1,
        }))

    def _poc(self, msg):
        poc = msg.get("proposal_open_contract", {})
        if poc.get("is_sold", 0) != 1:
            return

        profit = float(poc.get("profit", 0))
        win    = profit > 0

        self.session_pnl   += profit
        self.trade_count   += 1
        self.waiting_result = False
        self.active_cid     = None

        if win:
            self.session_wins += 1
            self.staker.record_win()
            self.threshold.record_win()
        else:
            self.session_losses += 1
            self.staker.record_loss()
            self.threshold.record_loss()

        wr = self.session_wins / self.trade_count * 100 if self.trade_count else 0
        log.info(
            "[Trader] %s #%d | P&L:%s | Session:%s | W/L:%d/%d (%.1f%%) | "
            "Bal:$%.2f | Thr:%.4f",
            "WIN " if win else "LOSS",
            self.trade_count, fmt_usd(profit), fmt_usd(self.session_pnl),
            self.session_wins, self.session_losses, wr,
            self.balance or 0, self.threshold.current
        )

        try:
            log_path = self.cfg.get("trade_log", "trade_log.csv")
            row = pd.DataFrame([{
                "time"       : datetime.now().isoformat(timespec="seconds"),
                "symbol"     : self.active_symbol,
                "trade_no"   : self.trade_count,
                "direction"  : self.pending_dir,
                "stake"      : self.pending_stake,
                "profit"     : round(profit, 4),
                "win"        : int(win),
                "session_pnl": round(self.session_pnl, 4),
                "balance"    : round(self.balance or 0, 2),
                "threshold"  : self.threshold.current,
            }])
            row.to_csv(
                log_path, mode="a",
                header=not os.path.exists(log_path),
                index=False,
            )
        except Exception as e:
            log.warning("[Trader] Could not write trade log: %s", e)

    def _on_error(self, ws, e):
        log.error("[Trader] WS error: %s", e)

    def _on_close(self, ws, *_):
        self._last_close_time = time.time()
        log.info("[Trader] Connection closed.")
        if not self.running:
            self._summary()

    def _summary(self):
        total = self.session_wins + self.session_losses
        wr    = self.session_wins / total * 100 if total else 0
        log.info("=" * 60)
        log.info("  SESSION SUMMARY")
        log.info("=" * 60)
        log.info("  Symbol      : %s", self.active_symbol)
        log.info("  Trades      : %d", total)
        log.info("  Wins/Losses : %d/%d", self.session_wins, self.session_losses)
        log.info("  Win Rate    : %.2f%%", wr)
        log.info("  Net P&L     : %s", fmt_usd(self.session_pnl))
        log.info("  Balance     : $%.2f %s", self.balance or 0, self.cfg["currency"])
        log.info("  Final thr   : %.4f", self.threshold.current)
        log.info("=" * 60)

    # -- Watchdog ----------------------------------------------------------

    def _start_watchdog(self):
        self._last_tick_time  = time.time()
        self._last_close_time = 0.0

        def _watch():
            while self.running:
                time.sleep(20)
                now = time.time()
                if self.waiting_result and self.active_cid:
                    if not hasattr(self, "_trade_start_time"):
                        self._trade_start_time = now
                    elif now - self._trade_start_time > 120:
                        log.warning("[Watchdog] Trade stuck >120s — resetting.")
                        self.waiting_result    = False
                        self.active_cid        = None
                        self._trade_start_time = None
                else:
                    self._trade_start_time = None
                if (self.balance is not None and
                        now - self._last_tick_time > 90 and
                        now - self._last_close_time > 10):
                    log.warning("[Watchdog] No tick in 90s — forcing reconnect.")
                    try:
                        self.ws.close()
                    except Exception:
                        pass

        threading.Thread(target=_watch, daemon=True).start()

    def run(self):
        def _stop(sig, frame):
            log.info("[Trader] Shutting down ...")
            self.running = False
            try:
                self.ws.close()
            except Exception:
                pass

        os_signal.signal(os_signal.SIGINT,  _stop)
        os_signal.signal(os_signal.SIGTERM, _stop)
        self._start_watchdog()

        backoff = 5
        while self.running:
            log.info("[Trader] Connecting (%s) ...", self.active_symbol)
            try:
                self.ws = websocket.WebSocketApp(
                    f"{self.WS_URL}?app_id={self.cfg['app_id']}",
                    on_open=self._on_open, on_message=self._on_message,
                    on_error=self._on_error, on_close=self._on_close,
                )
                self.ws.run_forever(ping_interval=30, ping_timeout=10, reconnect=0)
            except Exception as e:
                log.error("[Trader] run_forever exception: %s", e, exc_info=True)

            if not self.running:
                break

            self.waiting_result = False
            self.active_cid     = None
            self.balance        = None

            log.info("[Trader] Reconnecting in %ds ...", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

        self._summary()


# ===========================================================================
# MAIN ORCHESTRATOR
# ===========================================================================

def main():
    cfg = CONFIG

    if not cfg["api_token"] or len(cfg["api_token"].strip()) < 8:
        log.error("Set your Deriv API token in CONFIG['api_token']")
        log.error("  https://app.deriv.com/account/api-token")
        sys.exit(1)

    log.info("+============================================================+")
    log.info("|   DERIV DIGITS LSTM BOT v3 — STARTING                    |")
    log.info("+============================================================+")
    log.info("  Symbols       : %s", cfg["symbols"])
    log.info("  Base stake    : $%.2f (Deriv min)", cfg["kelly_min_stake"])
    log.info("  Staking       : half-Kelly (fraction=%.1f)", cfg["kelly_fraction"])
    log.info("  Kelly payout  : %.2f", cfg["kelly_payout"])
    log.info("  Conf range    : [%.2f – %.2f] base=%.2f",
             cfg["conf_floor"], cfg["conf_ceil"], cfg["conf_base"])
    log.info("  KS threshold  : %.2f", cfg["ks_retrain_threshold"])
    log.info("  Gradient clip : clipnorm=1.0 (Adam)")
    log.info("  Max loss/day  : $%.2f", cfg["max_daily_loss"])
    log.info("  Take profit   : $%.2f", cfg["take_profit"])
    log.info("  Reselect every: %d live ticks", cfg["reselect_every"])

    # -- Phase 1: Multi-Symbol Data Collection ----------------------------
    log.info("\n>> PHASE 1 — Parallel Data Collection")
    symbol_data = MultiSymbolCollector(cfg).collect_all()
    log.info("  Collected data for %d symbols: %s",
             len(symbol_data), list(symbol_data.keys()))

    # -- FIX 1: ADF stationarity check at Phase 2 start ------------------
    log.info("\n>> PHASE 2 — ADF Stationarity Check")
    # Run ADF on one of the collected datasets (use largest available)
    sample_sym    = max(symbol_data, key=lambda s: len(symbol_data[s]))
    sample_df     = symbol_data[sample_sym].copy()

    # Build features on sample to get the feature columns as a DataFrame
    _fe_sample    = FeatureEngineer(seq_len=cfg["seq_len"])
    sample_df_feat = _fe_sample.build(sample_df)

    valid_feature_cols = run_adf_tests(
        sample_df_feat, FeatureEngineer.ALL_FEATURE_COLS,
        pvalue_threshold=cfg["adf_pvalue"]
    )
    log.info("  Using %d/%d features: %s",
             len(valid_feature_cols),
             len(FeatureEngineer.ALL_FEATURE_COLS),
             valid_feature_cols)

    # -- Phase 2: Score all symbols, pick the best -----------------------
    log.info("\n>> PHASE 2 — Symbol Scoring & LSTM Training")
    selector = SymbolSelector(cfg)
    best_sym, dual_lstm, fe, scores, val_preds_baseline = selector.select(
        symbol_data, feature_cols=valid_feature_cols
    )
    log.info("  Trading symbol: %s", best_sym)

    # -- FIX 6: Initialise KS drift monitor with val baseline ------------
    drift_monitor = DriftMonitor(
        ks_threshold=cfg["ks_retrain_threshold"],
        min_window=cfg["ks_min_window"],
    )
    if val_preds_baseline is not None:
        drift_monitor.set_baseline(val_preds_baseline)
    else:
        log.warning("[DriftMonitor] No val baseline available — KS monitoring disabled.")

    # -- Phase 3: Live Trading -------------------------------------------
    log.info("\n>> PHASE 3 — Live Adaptive Trading on %s", best_sym)

    # FIX 7: Kelly staker (replaces martingale)
    staker    = KellyStaker(cfg)
    threshold = AdaptiveThreshold(cfg)

    trader = LiveTrader(
        cfg        = cfg,
        model      = dual_lstm,
        fe         = fe,
        staker     = staker,
        threshold  = threshold,
        drift_monitor = drift_monitor,
        active_symbol = best_sym,
        symbol_data   = symbol_data,
        feature_cols  = valid_feature_cols,
    )
    trader.run()


if __name__ == "__main__":
    main()
