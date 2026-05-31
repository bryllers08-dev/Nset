"""
+========================================================================+
|   DERIV DIGITS LSTM BOT - ADAPTIVE AUTONOMOUS TRADING SYSTEM        |
|                                                                      |
|   Phase 1 : Parallel data collection across R_10/25/50/75/100       |
|   Phase 2 : Dual-head LSTM training + symbol selection              |
|             - Head A -> P(even)         [parity prediction]          |
|             - Head B -> expiry (1-5t)   [duration prediction]        |
|             - LSTM scores all symbols, picks the best one            |
|   Phase 3 : Live trading on the chosen symbol                       |
|             - Dynamic confidence threshold (tightens on cold streak) |
|             - Dynamic expiry chosen by model per tick               |
|             - Martingale x2.1 after 2 consecutive losses            |
|             - Symbol re-evaluated every 3000 live ticks              |
+========================================================================+

 Requirements:
   pip install numpy pandas scikit-learn tensorflow websocket-client

 Usage:
   python deriv_lstm_bot.py

 Set your Deriv API token in CONFIG["api_token"] before running.
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
    Input, Softmax
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import websocket


# -- Logging ------------------------------------------------------------------
import io
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
# CONFIGURATION
# ===========================================================================


# ===========================================================================
# PERSISTENT STORAGE ROOT
# ===========================================================================
# On Railway: set the DATA_DIR environment variable to the path of your
# mounted Volume (e.g. /app/data).  All tick CSVs, trained models, and
# the trade log will be written there and survive redeployments.
#
# Locally: leave DATA_DIR unset and everything lands in ./bot_data/
#
DATA_DIR = os.environ.get("DATA_DIR", "bot_data")
os.makedirs(DATA_DIR, exist_ok=True)


CONFIG = {
    # -- Deriv API ----------------------------------------------------------
    "app_id"        : 1089,
    "api_token"     : os.environ.get("DERIV_API_TOKEN", "3nMoTkW49VHJqhH"),

    # All symbols to collect and score. The LSTM picks the best one.
    "symbols"       : ["R_10", "R_25", "R_50", "R_75", "R_100"],

    # Re-evaluate and potentially switch symbol every N live ticks.
    "reselect_every": 3000,

    # -- Data collection ---------------------------------------------------
    "collect_hours" : 2,
    "data_dir"      : os.path.join(DATA_DIR, "tick_data"),   # per-symbol CSVs
    "min_ticks"     : 3000,

    # -- Walk-forward validation -------------------------------------------
    # n folds, each with an OOS window of wf_oos_size sequences.
    # More folds = more robust score estimate, but longer Phase 2 time.
    "wf_folds"      : 5,
    "wf_oos_size"   : 600,

    # -- Online scaler -----------------------------------------------------
    # How many live ticks the scaler's EMA window covers.
    # Smaller = adapts faster but noisier. Larger = more stable.
    "scaler_window" : 10_000,

    # -- Model -------------------------------------------------------------
    "seq_len"       : 50,
    "epochs"        : 80,
    "batch_size"    : 64,
    "learning_rate" : 0.001,
    "dropout"       : 0.3,
    "model_file"    : os.path.join(DATA_DIR, "lstm_model_v2"),  # SavedModel dir

    # Expiry options the model can choose from (in ticks)
    "expiry_choices": [1, 2, 3, 4, 5],

    # -- Dynamic confidence threshold --------------------------------------
    # The AdaptiveThreshold engine adjusts these automatically.
    # These are the hard floor/ceiling -- model can't go outside this range.
    "conf_floor"    : 0.57,   # Never trade below this confidence
    "conf_ceil"     : 0.80,   # Never require above this (too restrictive)
    "conf_base"     : 0.60,   # Starting / resting threshold

    # Threshold tightens by this amount per consecutive loss
    "conf_loss_step": 0.02,
    # Threshold relaxes by this amount per consecutive win
    "conf_win_step" : 0.01,

    # -- Trading -----------------------------------------------------------
    "base_stake"    : 0.35,
    "currency"      : "USD",

    # -- Martingale --------------------------------------------------------
    "martingale_mult"      : 1.75,
    "martingale_after_loss": 2,
    "max_martingale_steps" : 4,

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
# PHASE 1 -- MULTI-SYMBOL DATA COLLECTOR
# ===========================================================================

class SymbolCollector:
    """Collects ticks for a single symbol. Runs on its own thread."""
    WS_URL = "wss://ws.binaryws.com/websockets/v3"

    def __init__(self, symbol: str, cfg: dict, done_event: threading.Event):
        self.symbol     = symbol
        self.cfg        = cfg
        self.done       = done_event
        self.ticks      = []
        self.end_time   = None

    def _on_open(self, ws):
        self.end_time = datetime.now() + timedelta(hours=self.cfg["collect_hours"])
        log.info(f"[Collector/{self.symbol}] Started -- collecting until "
                 f"{self.end_time.strftime('%H:%M:%S')} ...")
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
            log.info(f"[Collector/{self.symbol}] {n} ticks | {mins}m remaining")
        if datetime.now() >= self.end_time:
            ws.close()

    def _on_error(self, ws, e):
        log.error(f"[Collector/{self.symbol}] WS error: {e}")

    def _on_close(self, ws, *_):
        os.makedirs(self.cfg["data_dir"], exist_ok=True)
        path = os.path.join(self.cfg["data_dir"], f"ticks_{self.symbol}.csv")
        df = pd.DataFrame(self.ticks)
        df.to_csv(path, index=False)
        log.info(f"[Collector/{self.symbol}] Saved {len(df)} ticks -> {path}")
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
                    log.error(f"[Collector/{self.symbol}] Exception: {e}")
                if self.done.is_set():
                    break
                log.info(f"[Collector/{self.symbol}] Reconnecting in {backoff}s ...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
        threading.Thread(target=_run_with_retry, daemon=True).start()


class MultiSymbolCollector:
    """
    Spawns one SymbolCollector per symbol concurrently and waits for
    all of them to finish before returning the collected DataFrames.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def collect_all(self) -> dict:
        """
        Returns a dict  {symbol: DataFrame}  for every symbol that
        passed the min_ticks check. Symbols with too few ticks are
        logged and skipped.
        """
        symbols  = self.cfg["symbols"]
        events   = {s: threading.Event() for s in symbols}
        collectors = {}

        for sym in symbols:
            path = os.path.join(self.cfg["data_dir"], f"ticks_{sym}.csv")
            if os.path.exists(path):
                log.info(f"[Collector/{sym}] Reusing existing file: {path}")
                events[sym].set()          # mark as already done
            else:
                c = SymbolCollector(sym, self.cfg, events[sym])
                collectors[sym] = c
                c.start()

        # Block until every symbol's event fires
        for sym, ev in events.items():
            ev.wait()

        results = {}
        for sym in symbols:
            path = os.path.join(self.cfg["data_dir"], f"ticks_{sym}.csv")
            try:
                df = pd.read_csv(path)
                if len(df) < self.cfg["min_ticks"]:
                    log.warning(f"[Collector/{sym}] Only {len(df)} ticks "
                                f"(need {self.cfg['min_ticks']}) -- skipping.")
                else:
                    results[sym] = df
                    log.info(f"[Collector/{sym}] {len(df)} ticks loaded.")
            except Exception as e:
                log.error(f"[Collector/{sym}] Could not load CSV: {e}")

        if not results:
            log.error("[Collector] No symbol had enough ticks. Aborting.")
            sys.exit(1)

        return results


# ===========================================================================
# PHASE 2 -- ONLINE SCALER  (fixes scaler drift)
# ===========================================================================

class OnlineScaler:
    """
    Welford one-pass running mean/variance scaler.

    fit()          -- compute stats from a training array (same API as
                      StandardScaler.fit_transform, but stores params).
    transform()    -- scale a batch using stored training stats.
    update(vec)    -- fold one live feature vector into the running stats
                      so the scaler adapts gradually to distribution shift.
    check_drift()  -- compare a live vector against training stats and
                      return a per-feature z-score; high values signal
                      that the live distribution has moved out of range.

    The update window is bounded: once n_seen > max_window the oldest
    contribution is down-weighted so the scaler tracks the recent regime
    rather than the entire history.
    """

    def __init__(self, max_window: int = 10_000):
        self.max_window  = max_window
        self.n_seen      = 0
        self.mean_       = None
        self.var_        = None     # biased variance
        # Training stats kept separately for drift detection
        self.train_mean_ = None
        self.train_std_  = None

    # -- Batch fit (call once on training data) ----------------------------

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X (N, F) and return scaled copy. Stores training stats."""
        self.train_mean_ = X.mean(axis=0)
        self.train_std_  = X.std(axis=0) + 1e-8
        # Seed the online stats from training data
        self.mean_ = self.train_mean_.copy()
        self.var_  = (self.train_std_ ** 2)
        self.n_seen = len(X)
        return (X - self.train_mean_) / self.train_std_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale using current (possibly updated) stats."""
        std = np.sqrt(self.var_) + 1e-8
        return (X - self.mean_) / std

    # -- Single-vector live update (call on every incoming tick) -----------

    def update(self, vec: np.ndarray):
        """
        Welford incremental update. When n_seen > max_window we use an
        exponential decay (alpha = 1/max_window) so older observations
        contribute less -- keeps the scaler tracking the live regime.
        """
        if self.mean_ is None:
            self.mean_  = vec.copy()
            self.var_   = np.zeros_like(vec)
            self.n_seen = 1
            return

        self.n_seen += 1
        if self.n_seen <= self.max_window:
            # Standard Welford
            delta       = vec - self.mean_
            self.mean_ += delta / self.n_seen
            self.var_  += delta * (vec - self.mean_)   # M2 accumulator
            # Convert M2 to variance on the fly
            if self.n_seen > 1:
                self.var_ = self.var_ / (self.n_seen - 1)
        else:
            # EMA regime: alpha = 1/max_window
            alpha      = 1.0 / self.max_window
            delta      = vec - self.mean_
            self.mean_ = self.mean_ + alpha * delta
            self.var_  = (1 - alpha) * (self.var_ + alpha * delta ** 2)

    # -- Drift detection ---------------------------------------------------

    def check_drift(self, vec: np.ndarray,
                    warn_threshold: float = 4.0) -> dict:
        """
        Compare a live vector to the *training* distribution.
        Returns {"drifted": bool, "max_z": float, "features": [int, ...]}.
        Drifted features have |z| > warn_threshold (default 4 sigma).
        """
        if self.train_mean_ is None:
            return {"drifted": False, "max_z": 0.0, "features": []}
        z        = np.abs((vec - self.train_mean_) / self.train_std_)
        drifted  = z > warn_threshold
        return {
            "drifted"  : bool(drifted.any()),
            "max_z"    : float(z.max()),
            "features" : list(np.where(drifted)[0]),
        }


# ===========================================================================
# PHASE 2 -- FEATURE ENGINEERING
# ===========================================================================

class FeatureEngineer:
    """
    9 features per tick.

    Expiry labels (fix: no lookahead bias)
    ---------------------------------------
    The old approach looked at future ticks to decide the "best" expiry,
    which leaks future information into training. The corrected approach
    uses only *lagged* realised parity runs: for each tick i we look
    backward at the previous max_expiry ticks and measure how consistent
    the parity was -- a long stable run suggests the current parity is
    likely to persist, so we assign a longer expiry label. This is purely
    causal and uses no future data.
    """

    FEATURE_COLS = [
        "digit", "label", "digit_norm",
        "freq_even_10", "freq_even_30",
        "streak", "digit_freq_norm",
        "price_delta", "volatility_10",
    ]
    EXPIRY_OPTIONS = [1, 2, 3, 4, 5]   # ticks

    # Drift warning: suppress repeated warnings within this many ticks
    _DRIFT_WARN_COOLDOWN = 200

    def __init__(self, seq_len: int, scaler_window: int = 10_000):
        self.seq_len          = seq_len
        self.scaler           = OnlineScaler(max_window=scaler_window)
        self._drift_warn_tick = 0    # cooldown counter for drift warnings
        self._live_tick_count = 0

    # -- Feature computation -----------------------------------------------

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        df["freq_even_10"]    = df["label"].rolling(10, min_periods=1).mean()
        df["freq_even_30"]    = df["label"].rolling(30, min_periods=1).mean()
        df["digit_norm"]      = df["digit"] / 9.0
        df["price_delta"]     = df["price"].diff().fillna(0)
        df["volatility_10"]   = df["digit"].rolling(10, min_periods=1).std().fillna(0)
        df["digit_freq_norm"] = (
            df["digit"].rolling(20, min_periods=1)
            .apply(lambda x: np.sum(x == x.iloc[-1]) / len(x), raw=False)
        )
        streaks, s, prev = [], 0, None
        for p in df["label"]:
            s = (s+1 if p==1 else s-1) if p==prev else (1 if p==1 else -1)
            streaks.append(s); prev = p
        df["streak"] = streaks

        df["expiry_label"] = self._build_expiry_labels(df)
        return df

    def _build_expiry_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Causal expiry label -- no lookahead.

        For each tick i we look at the PREVIOUS max_expiry ticks and
        measure parity consistency (fraction of ticks matching the most
        recent parity). High consistency -> assign longer expiry class.

        Mapping:
          consistency >= 0.80  -> expiry index 4  (5t)
          consistency >= 0.65  -> expiry index 3  (4t)
          consistency >= 0.55  -> expiry index 2  (3t)
          consistency >= 0.45  -> expiry index 1  (2t)
          otherwise            -> expiry index 0  (1t)
        """
        labels    = df["label"].values
        max_exp   = self.EXPIRY_OPTIONS[-1]   # 5
        thresholds = [0.80, 0.65, 0.55, 0.45]  # descending
        result    = np.zeros(len(labels), dtype=np.int32)

        for i in range(len(labels)):
            if i == 0:
                result[i] = 0
                continue
            window     = labels[max(0, i - max_exp) : i]
            cur_parity = labels[i]
            consistency = float(np.sum(window == cur_parity) / len(window))
            assigned   = 0
            for exp_idx, thr in enumerate(thresholds):
                if consistency >= thr:
                    assigned = len(thresholds) - exp_idx   # 4, 3, 2, 1
                    break
            result[i] = assigned

        return result

    # -- Sequence builder --------------------------------------------------

    def make_sequences(self, df: pd.DataFrame):
        """
        Returns:
          X          -- (N, seq_len, features)
          y_parity   -- (N,)  binary even/odd
          y_expiry   -- (N, 5) one-hot expiry class

        The scaler is fit here on the full training frame so training
        stats are stored for later drift detection.
        """
        feat   = df[self.FEATURE_COLS].values.astype(np.float32)
        feat   = self.scaler.fit_transform(feat)   # stores train stats
        parity = df["label"].values.astype(np.int32)
        expiry = df["expiry_label"].values.astype(np.int32)

        X, yp, ye = [], [], []
        for i in range(self.seq_len, len(feat)):
            X.append(feat[i - self.seq_len : i])
            yp.append(parity[i])
            ye.append(expiry[i])

        X  = np.array(X,  dtype=np.float32)
        yp = np.array(yp, dtype=np.float32)
        ye = tf.keras.utils.to_categorical(ye, num_classes=5).astype(np.float32)
        return X, yp, ye

    def transform_one(self, vec: np.ndarray) -> np.ndarray:
        """
        Scale one live feature vector, update the online scaler, and
        emit a drift warning if the vector is far outside training stats.
        """
        self._live_tick_count += 1

        # 1. Drift check against frozen training distribution
        drift = self.scaler.check_drift(vec)
        if drift["drifted"]:
            ticks_since_warn = (self._live_tick_count - self._drift_warn_tick)
            if ticks_since_warn >= self._DRIFT_WARN_COOLDOWN:
                self._drift_warn_tick = self._live_tick_count
                log.warning(
                    f"[Scaler] Distribution drift detected! "
                    f"max_z={drift['max_z']:.2f} on feature(s) "
                    f"{drift['features']} -- model inputs are out of "
                    f"training range. Consider triggering a reselect."
                )

        # 2. Update the online scaler with this live observation
        self.scaler.update(vec)

        # 3. Scale using the (now updated) running stats
        return self.scaler.transform(vec.reshape(1, -1))[0]


# ===========================================================================
# PHASE 2 -- DUAL-HEAD LSTM MODEL
# ===========================================================================

class DualHeadLSTM:
    """
    Shared LSTM backbone with two output heads:
      - parity_out  : sigmoid  -> P(even)          [0..1]
      - expiry_out  : softmax  -> P(1t|2t|3t|4t|5t) [5-class]

    The model is trained jointly; the shared trunk learns features
    useful for BOTH predicting the outcome AND the best holding period.
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

        # -- Head A -- parity -----------------------------------------------
        pa = Dense(16, activation="relu",    name="parity_hidden")(shared)
        pa = Dense(1,  activation="sigmoid", name="parity_out")(pa)

        # -- Head B -- expiry -----------------------------------------------
        ex = Dense(16, activation="relu",    name="expiry_hidden")(shared)
        ex = Dense(5,  activation="softmax", name="expiry_out")(ex)

        model = Model(inputs=inp, outputs=[pa, ex], name="DualHeadLSTM")
        model.compile(
            optimizer=Adam(self.cfg["learning_rate"]),
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
                          restore_best_weights=True, verbose=1,
                          mode="max"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=6, min_lr=1e-6, verbose=1),
        ]
        history = self.model.fit(
            X_tr,
            {"parity_out": yp_tr, "expiry_out": ye_tr},
            validation_data=(
                X_val, {"parity_out": yp_val, "expiry_out": ye_val}
            ),
            epochs=self.cfg["epochs"],
            batch_size=self.cfg["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )
        # Save
        self.model.save(self.cfg["model_file"])
        log.info(f"[Trainer] Model saved -> {self.cfg['model_file']}")
        return history

    def predict(self, seq: np.ndarray):
        """
        seq : (seq_len, features)
        Returns:
          p_even  : float  -- probability of EVEN
          expiry  : int    -- chosen duration in ticks (1..5)
          exp_conf: float  -- confidence in expiry choice
        """
        p_even_arr, exp_arr = self.model.predict(
            seq[np.newaxis, ...], verbose=0
        )
        p_even   = float(p_even_arr[0, 0])
        exp_probs= exp_arr[0]                       # shape (5,)
        best_idx = int(np.argmax(exp_probs))
        exp_conf = float(exp_probs[best_idx])
        expiry   = self.EXPIRY_TICKS[best_idx]
        return p_even, expiry, exp_conf

    @classmethod
    def load_saved(cls, cfg):
        obj = cls.__new__(cls)
        obj.cfg   = cfg
        obj.model = load_model(cfg["model_file"])
        log.info(f"[Trainer] Loaded model <- {cfg['model_file']}")
        return obj


# ===========================================================================
# SYMBOL SELECTOR
# ===========================================================================

class SymbolSelector:
    """
    Trains (or reuses) one DualHeadLSTM per symbol, scores each on its
    last 3 000 ticks using parity-head accuracy, and returns the symbol
    with the highest score along with its fitted FeatureEngineer.

    The selection logic is intentionally kept separate so it can be
    called both at startup (Phase 2) and periodically during live
    trading (every reselect_every ticks).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def select(self, symbol_data: dict) -> tuple:
        """
        Args:
            symbol_data: {symbol: DataFrame}  -- full tick history per symbol

        Returns:
            (best_symbol, best_model, best_fe, scores_dict)
              best_symbol : str
              best_model  : DualHeadLSTM
              best_fe     : FeatureEngineer  (already fit on best_symbol data)
              scores_dict : {symbol: float}  mean OOS parity accuracy across folds
        """
        cfg      = self.cfg
        seq_len  = cfg["seq_len"]
        n_folds  = cfg.get("wf_folds", 5)
        # Each OOS fold covers this many sequences
        oos_size = cfg.get("wf_oos_size", 600)

        scores  = {}
        trained = {}   # symbol -> (model, fe)

        log.info(f"[Selector] Walk-forward validation: {n_folds} folds, "
                 f"{oos_size} OOS sequences each ...")

        for sym, df_full in symbol_data.items():
            fe      = FeatureEngineer(
                seq_len=seq_len,
                scaler_window=cfg.get("scaler_window", 10_000),
            )
            df_feat = fe.build(df_full)
            X, yp, ye = fe.make_sequences(df_feat)

            min_needed = seq_len + n_folds * oos_size + oos_size
            if len(X) < min_needed:
                log.warning(f"[Selector/{sym}] Not enough sequences "
                            f"({len(X)} < {min_needed}) -- skipping.")
                continue

            # -- Walk-forward folds ----------------------------------------
            # Layout: [burn-in | fold-1-train | fold-1-OOS | ... | fold-N-OOS]
            # Each fold trains on everything up to its OOS window, then
            # evaluates on the OOS window. No future data ever leaks in.
            fold_accs = []
            # The earliest the first OOS window can end
            first_oos_end = len(X) - (n_folds - 1) * oos_size
            oos_ends = [first_oos_end + k * oos_size for k in range(n_folds)]
            # Clamp last fold to actual data end
            oos_ends[-1] = min(oos_ends[-1], len(X))

            model_path = os.path.join(
                cfg.get("data_dir", "tick_data"),
                f"model_{sym}"
            )

            for fold_idx, oos_end in enumerate(oos_ends):
                oos_start = oos_end - oos_size
                if oos_start <= seq_len:
                    continue   # not enough training data before this fold

                X_tr  = X[:oos_start]
                yp_tr = yp[:oos_start]
                ye_tr = ye[:oos_start]
                X_oos  = X[oos_start:oos_end]
                yp_oos = yp[oos_start:oos_end]

                # Val split from tail of training block
                val_size = max(1, int(len(X_tr) * 0.12))
                X_val  = X_tr[-val_size:];  yp_val = yp_tr[-val_size:]
                ye_val = ye_tr[-val_size:]
                X_tr   = X_tr[:-val_size];  yp_tr  = yp_tr[:-val_size]
                ye_tr  = ye_tr[:-val_size]

                n_feat = X.shape[2]

                # Only load a cached model on the FINAL fold (used for live
                # trading). Intermediate folds always retrain to get honest
                # OOS estimates.
                is_last_fold = (fold_idx == len(oos_ends) - 1)
                if is_last_fold and os.path.exists(model_path):
                    log.info(f"[Selector/{sym}] Fold {fold_idx+1}/{n_folds} "
                             "-- loading cached final model.")
                    dual = DualHeadLSTM.load_saved(
                        {**cfg, "model_file": model_path}
                    )
                else:
                    log.info(f"[Selector/{sym}] Fold {fold_idx+1}/{n_folds} "
                             f"-- training on {len(X_tr)} sequences ...")
                    fold_model_path = (model_path if is_last_fold
                                       else model_path + f"_fold{fold_idx}")
                    dual = DualHeadLSTM(
                        seq_len, n_feat,
                        {**cfg, "model_file": fold_model_path}
                    )
                    dual.train(X_tr, yp_tr, ye_tr, X_val, yp_val, ye_val)

                pa_pred, _ = dual.model.predict(X_oos, verbose=0)
                pa_bin     = (pa_pred.flatten() > 0.5).astype(int)
                fold_acc   = float(np.mean(pa_bin == yp_oos.astype(int)))
                fold_accs.append(fold_acc)
                log.info(f"[Selector/{sym}] Fold {fold_idx+1} OOS accuracy: "
                         f"{fold_acc:.4f} ({fold_acc*100:.2f}%)")

                # Keep the final fold's model for live trading
                if is_last_fold:
                    trained[sym] = (dual, fe)

            if not fold_accs:
                log.warning(f"[Selector/{sym}] No valid folds -- skipping.")
                continue

            mean_acc  = float(np.mean(fold_accs))
            std_acc   = float(np.std(fold_accs))
            scores[sym] = mean_acc
            log.info(f"[Selector/{sym}] Mean OOS accuracy: "
                     f"{mean_acc:.4f} +/- {std_acc:.4f} "
                     f"over {len(fold_accs)} folds")

        if not scores:
            log.error("[Selector] No symbol could be scored. Aborting.")
            sys.exit(1)

        best_sym  = max(scores, key=scores.__getitem__)
        best_acc  = scores[best_sym]
        best_model, best_fe = trained[best_sym]

        log.info("+------------------------------------------------------------+")
        log.info(f"  Symbol ranking:")
        for sym, acc in sorted(scores.items(), key=lambda x: -x[1]):
            marker = " <-- SELECTED" if sym == best_sym else ""
            log.info(f"    {sym:6s}  accuracy={acc:.4f} ({acc*100:.2f}%){marker}")
        log.info(f"  Selected: {best_sym}  (accuracy={best_acc:.4f})")
        log.info("+------------------------------------------------------------+")

        return best_sym, best_model, best_fe, scores


# ===========================================================================
# ADAPTIVE CONFIDENCE THRESHOLD
# ===========================================================================

class AdaptiveThreshold:
    """
    Starts at conf_base.
    After each loss  -> threshold += loss_step  (requires higher confidence)
    After each win   -> threshold -= win_step   (relaxes toward base)
    Always clamped to [conf_floor, conf_ceil].

    Also factors in the model's own expiry confidence:
    If the model is uncertain about the expiry, we require a slightly
    higher parity confidence before trading.
    """

    def __init__(self, cfg: dict):
        self.base       = cfg["conf_base"]
        self.floor      = cfg["conf_floor"]
        self.ceil       = cfg["conf_ceil"]
        self.loss_step  = cfg["conf_loss_step"]
        self.win_step   = cfg["conf_win_step"]
        self.threshold  = cfg["conf_base"]
        self.consec_loss= 0
        self.consec_win = 0

    @property
    def current(self) -> float:
        return round(self.threshold, 4)

    def effective(self, expiry_conf: float) -> float:
        """
        Adjust threshold upward when expiry confidence is low.
        Low expiry confidence (<0.4) adds up to +0.03 to threshold.
        """
        penalty = max(0.0, (0.4 - expiry_conf) * 0.15)
        return min(self.ceil, self.threshold + penalty)

    def record_win(self):
        self.consec_loss = 0
        self.consec_win += 1
        self.threshold   = max(
            self.floor,
            self.threshold - self.win_step
        )
        log.info(f"[Threshold] WIN  -> threshold relaxed to {self.threshold:.4f}")

    def record_loss(self):
        self.consec_win  = 0
        self.consec_loss += 1
        self.threshold    = min(
            self.ceil,
            self.threshold + self.loss_step
        )
        log.info(
            f"[Threshold] LOSS ({self.consec_loss} consec) -> "
            f"threshold tightened to {self.threshold:.4f}"
        )

    def reset(self):
        self.threshold   = self.base
        self.consec_loss = 0
        self.consec_win  = 0


# ===========================================================================
# MARTINGALE STAKER
# ===========================================================================

class MartingaleStaker:
    def __init__(self, base, multiplier, trigger_after, max_steps):
        self.base          = base
        self.multiplier    = multiplier
        self.trigger_after = trigger_after
        self.max_steps     = max_steps
        self.consec_losses = 0
        self.current_stake = base

    def next_stake(self) -> float:
        return round(self.current_stake, 2)

    def record_win(self):
        self.consec_losses = 0
        self.current_stake = self.base
        log.info(f"[Martingale] WIN -> stake reset to ${self.base}")

    def record_loss(self):
        self.consec_losses += 1
        if self.consec_losses >= self.trigger_after:
            steps = min(
                self.consec_losses - self.trigger_after + 1,
                self.max_steps
            )
            self.current_stake = round(
                self.base * (self.multiplier ** steps), 2
            )
            log.info(
                f"[Martingale] {self.consec_losses} consec losses -> "
                f"stake ${self.current_stake} "
                f"(x{self.multiplier}^{steps})"
            )
        else:
            log.info(
                f"[Martingale] {self.consec_losses} loss(es) -- "
                f"martingale not yet active"
            )


# ===========================================================================
# PHASE 3 -- LIVE TRADER
# ===========================================================================

class LiveTrader:
    WS_URL = "wss://ws.binaryws.com/websockets/v3"

    def __init__(self, cfg, model: DualHeadLSTM,
                 fe: FeatureEngineer,
                 staker: MartingaleStaker,
                 threshold: AdaptiveThreshold,
                 active_symbol: str,
                 symbol_data: dict):
        self.cfg           = cfg
        self.model         = model
        self.fe            = fe
        self.staker        = staker
        self.threshold     = threshold
        self.active_symbol = active_symbol
        self.symbol_data   = symbol_data   # {sym: DataFrame} kept for reselection
        self.selector      = SymbolSelector(cfg)
        self.ws            = None

        # Rolling buffers
        self.tick_buf = deque(maxlen=2000)
        self.seq_buf  = deque(maxlen=cfg["seq_len"])

        # Trade state
        self.active_cid     = None
        self.pending_dir    = None
        self.pending_stake  = None
        self.waiting_result = False

        # Session stats
        self.session_pnl    = 0.0
        self.session_wins   = 0
        self.session_losses = 0
        self.trade_count    = 0
        self.balance        = None
        self.running        = True

        # Symbol reselection counter
        self.live_tick_count   = 0
        self.reselect_every    = cfg["reselect_every"]
        self.reselecting       = False   # guard against concurrent reselects
        # Watchdog timestamps (set in _start_watchdog, pre-init here)
        self._last_tick_time   = time.time()
        self._last_close_time  = 0.0

    # -- Feature helpers ---------------------------------------------------

    def _feature_vec(self) -> np.ndarray:
        buf = list(self.tick_buf)
        n   = len(buf)
        d   = buf[-1]["digit"]
        lbl = buf[-1]["label"]

        labels10 = [b["label"] for b in buf[max(0,n-10):]]
        labels30 = [b["label"] for b in buf[max(0,n-30):]]
        fe10, fe30 = np.mean(labels10), np.mean(labels30)

        streak, prev = 0, None
        for b in buf:
            p = b["label"]
            streak = (streak+1 if p==1 else streak-1) if p==prev else (1 if p==1 else -1)
            prev = p

        d20   = [b["digit"] for b in buf[max(0,n-20):]]
        dfreq = d20.count(d) / len(d20)

        pdelta = (buf[-1]["price"] - buf[-2]["price"]) if n >= 2 else 0.0
        d10    = [b["digit"] for b in buf[max(0,n-10):]]
        vol10  = float(np.std(d10)) if len(d10) > 1 else 0.0

        return np.array([
            d, lbl, d/9.0, fe10, fe30,
            streak, dfreq, pdelta, vol10
        ], dtype=np.float32)

    # -- Tick subscription helper ------------------------------------------

    def _subscribe_ticks(self, symbol: str):
        self.ws.send(json.dumps({"ticks": symbol, "subscribe": 1}))

    def _unsubscribe_ticks(self, symbol: str):
        self.ws.send(json.dumps({"forget_all": "ticks"}))

    # -- Periodic symbol reselection ---------------------------------------

    def _maybe_reselect(self):
        """
        Called every reselect_every live ticks. Merges the live tick buffer
        into the historical data for each symbol, re-scores, and switches
        symbol + model if a better one is found.
        Runs in a background thread so it doesn't block tick processing.
        """
        if self.reselecting or self.waiting_result:
            return
        self.reselecting = True

        def _run():
            log.info(f"[Selector] {self.reselect_every} live ticks reached -- "
                     "re-evaluating symbols ...")

            # Append live ticks (from active symbol) to its historical data
            if self.tick_buf:
                live_df = pd.DataFrame(list(self.tick_buf))
                if self.active_symbol in self.symbol_data:
                    self.symbol_data[self.active_symbol] = pd.concat(
                        [self.symbol_data[self.active_symbol], live_df],
                        ignore_index=True
                    )
                else:
                    self.symbol_data[self.active_symbol] = live_df

            try:
                new_sym, new_model, new_fe, scores = self.selector.select(
                    self.symbol_data
                )
            except Exception as e:
                log.error(f"[Selector] Re-selection failed: {e}", exc_info=True)
                self.reselecting = False
                return

            if new_sym != self.active_symbol:
                log.info(f"[Selector] Switching symbol: "
                         f"{self.active_symbol} -> {new_sym}")
                self._unsubscribe_ticks(self.active_symbol)
                self.active_symbol = new_sym
                self.model         = new_model
                self.fe            = new_fe
                # Clear buffers so we start fresh on the new symbol
                self.tick_buf.clear()
                self.seq_buf.clear()
                self.threshold.reset()
                self._subscribe_ticks(self.active_symbol)
                log.info(f"[Selector] Now trading {self.active_symbol}. "
                         "Refilling sequence buffer ...")
            else:
                log.info(f"[Selector] {self.active_symbol} remains best -- "
                         "no switch needed.")

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
            if   mtype == "authorize"               : self._auth(msg)
            elif mtype == "balance"                  : self._balance(msg)
            elif mtype == "tick"                     : self._tick(msg)
            elif mtype == "buy"                      : self._buy(msg)
            elif mtype == "proposal_open_contract"   : self._poc(msg)
            elif mtype == "error":
                log.error(f"[Trader] API: {msg['error']['message']}")
        except Exception as e:
            log.error(f"[Trader] Handler error: {e}", exc_info=True)

    def _auth(self, msg):
        if "error" in msg:
            log.error(f"[Trader] Auth failed: {msg['error']['message']}")
            self.running = False; return
        info = msg["authorize"]
        self.balance = float(info.get("balance", 0))
        log.info(f"[Trader] Logged in as {info.get('loginid','?')} | "
                 f"Balance: ${self.balance:.2f}")
        self.ws.send(json.dumps({"balance": 1, "subscribe": 1}))
        self._subscribe_ticks(self.active_symbol)
        log.info(f"[Trader] Streaming {self.active_symbol} -- filling sequence buffer ...")

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
        if self.live_tick_count >= self.reselect_every:
            self._maybe_reselect()

        if len(self.tick_buf) < 5:
            return

        # Update sequence buffer
        fvec   = self._feature_vec()
        scaled = self.fe.transform_one(fvec)
        self.seq_buf.append(scaled)

        if self.waiting_result or not self.running:
            return
        if len(self.seq_buf) < self.cfg["seq_len"]:
            remaining = self.cfg["seq_len"] - len(self.seq_buf)
            if remaining % 10 == 0:
                log.info(f"[Trader] Building buffer ... {remaining} ticks remaining")
            return

        # -- Risk guards ---------------------------------------------------
        if self.session_pnl <= -self.cfg["max_daily_loss"]:
            log.warning(f"[Trader] [STOP] Max daily loss (${self.cfg['max_daily_loss']}) hit.")
            self.running = False; self.ws.close(); return
        if self.trade_count >= self.cfg["max_trades_day"]:
            log.warning("[Trader] [STOP] Max trades reached.")
            self.running = False; self.ws.close(); return
        if self.session_pnl >= self.cfg["take_profit"]:
            log.info(f"[Trader] [TP] Take-profit! P&L: ${self.session_pnl:.2f}")
            self.running = False; self.ws.close(); return

        # -- Model inference -----------------------------------------------
        seq       = np.array(self.seq_buf, dtype=np.float32)
        p_even, expiry, exp_conf = self.model.predict(seq)

        # Dynamic threshold -- tightens on loss streak + expiry uncertainty
        thr = self.threshold.effective(exp_conf)

        if p_even >= thr:
            direction, trade_prob = "EVEN", p_even
        elif (1 - p_even) >= thr:
            direction, trade_prob = "ODD", 1 - p_even
        else:
            return   # Not confident enough -- sit this one out

        stake = self.staker.next_stake()

        log.info(
            f"[Trader] >> Signal: {direction:<4} | "
            f"P: {trade_prob:.3f} (thr={thr:.3f}) | "
            f"Expiry: {expiry}t (conf={exp_conf:.2f}) | "
            f"Stake: ${stake} | Threshold: {self.threshold.current:.4f}"
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
            log.error(f"[Trader] Buy error: {msg['error']['message']}")
            self.waiting_result = False; return
        self.active_cid = msg["buy"]["contract_id"]
        log.info(f"[Trader] Contract placed: {self.active_cid}")
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

        wr = (self.session_wins / self.trade_count * 100
              if self.trade_count else 0)

        log.info(
            f"[Trader] {'[WIN] WIN ' if win else '[LOSS] LOSS'} "
            f"#{self.trade_count} | "
            f"P&L: {fmt_usd(profit)} | "
            f"Session: {fmt_usd(self.session_pnl)} | "
            f"W/L: {self.session_wins}/{self.session_losses} ({wr:.1f}%) | "
            f"Threshold now: {self.threshold.current:.4f} | "
            f"Bal: ${self.balance:.2f}"
        )

        # -- Persist trade to CSV log --------------------------------------
        try:
            log_path = self.cfg.get("trade_log", "trade_log.csv")
            row = pd.DataFrame([{
                "time"      : datetime.now().isoformat(timespec="seconds"),
                "symbol"    : self.active_symbol,
                "trade_no"  : self.trade_count,
                "direction" : self.pending_dir,
                "stake"     : self.pending_stake,
                "profit"    : round(profit, 4),
                "win"       : int(win),
                "session_pnl": round(self.session_pnl, 4),
                "balance"   : round(self.balance, 2),
                "threshold" : self.threshold.current,
            }])
            row.to_csv(
                log_path,
                mode="a",
                header=not os.path.exists(log_path),
                index=False,
            )
        except Exception as e:
            log.warning(f"[Trader] Could not write trade log: {e}")

    def _on_error(self, ws, e):
        log.error(f"[Trader] WS error: {e}")

    def _on_close(self, ws, *_):
        self._last_close_time = time.time()
        log.info("[Trader] Connection closed.")
        if self.running:
            log.info("[Trader] Will attempt reconnect ...")
        else:
            self._summary()

    def _summary(self):
        total = self.session_wins + self.session_losses
        wr    = self.session_wins / total * 100 if total else 0
        log.info("=" * 60)
        log.info("  SESSION SUMMARY")
        log.info("=" * 60)
        log.info(f"  Symbol      : {self.active_symbol}")
        log.info(f"  Trades      : {total}")
        log.info(f"  Wins/Losses : {self.session_wins}/{self.session_losses}")
        log.info(f"  Win Rate    : {wr:.2f}%")
        log.info(f"  Net P&L     : {fmt_usd(self.session_pnl)}")
        log.info(f"  Balance     : ${self.balance:.2f} {self.cfg['currency']}")
        log.info(f"  Final thr   : {self.threshold.current:.4f}")
        log.info("=" * 60)

    # -- Heartbeat watchdog ------------------------------------------------

    def _start_watchdog(self):
        """
        Background thread that checks:
          1. Last tick age -- if no tick in 90s, force a reconnect.
          2. Stuck trade   -- if waiting_result for >120s, reset the flag
             so the bot doesn't freeze waiting for a contract that silently
             failed (e.g. connection dropped mid-trade).
        """
        self._last_tick_time  = time.time()
        self._last_close_time = 0.0

        def _watch():
            while self.running:
                time.sleep(20)
                now = time.time()

                # Stuck trade guard
                if self.waiting_result and self.active_cid:
                    if not hasattr(self, '_trade_start_time'):
                        self._trade_start_time = now
                    elif now - self._trade_start_time > 120:
                        log.warning("[Watchdog] Trade stuck >120s -- resetting state.")
                        self.waiting_result    = False
                        self.active_cid        = None
                        self._trade_start_time = None
                else:
                    self._trade_start_time = None

                # Tick timeout guard (only when connected)
                if (self.balance is not None and
                        now - self._last_tick_time > 90 and
                        now - self._last_close_time > 10):
                    log.warning("[Watchdog] No tick in 90s -- forcing reconnect.")
                    try:
                        self.ws.close()
                    except Exception:
                        pass

        threading.Thread(target=_watch, daemon=True).start()

    def run(self):
        import time as _time_mod  # ensure available inside closure

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

        MAX_BACKOFF = 60    # seconds
        backoff     = 5

        while self.running:
            log.info(f"[Trader] Connecting to Deriv ({self.active_symbol}) ...")
            try:
                self.ws = websocket.WebSocketApp(
                    f"{self.WS_URL}?app_id={self.cfg['app_id']}",
                    on_open=self._on_open, on_message=self._on_message,
                    on_error=self._on_error, on_close=self._on_close,
                )
                self.ws.run_forever(
                    ping_interval=30,
                    ping_timeout=10,
                    reconnect=0,       # we handle reconnect ourselves
                )
            except Exception as e:
                log.error(f"[Trader] run_forever exception: {e}", exc_info=True)

            if not self.running:
                break

            # Reset transient per-connection state
            self.waiting_result = False
            self.active_cid     = None
            self.balance        = None   # will be re-set on next auth

            log.info(f"[Trader] Reconnecting in {backoff}s ...")
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)

        self._summary()


# ===========================================================================
# MAIN ORCHESTRATOR
# ===========================================================================

def main():
    cfg = CONFIG

    if not cfg["api_token"] or len(cfg["api_token"].strip()) < 8:
        log.error("[LOSS]  Set your Deriv API token in CONFIG['api_token']")
        log.error("    https://app.deriv.com/account/api-token")
        sys.exit(1)

    log.info("+============================================================+")
    log.info("|   DERIV DIGITS LSTM BOT -- ADAPTIVE SYSTEM STARTING      |")
    log.info("+============================================================+")
    log.info(f"  Symbols       : {cfg['symbols']}")
    log.info(f"  Base Stake    : ${cfg['base_stake']}")
    log.info(f"  Martingale    : x{cfg['martingale_mult']} after "
             f"{cfg['martingale_after_loss']} losses")
    log.info(f"  Conf range    : [{cfg['conf_floor']} - {cfg['conf_ceil']}] "
             f"base={cfg['conf_base']}")
    log.info(f"  Expiry range  : {cfg['expiry_choices']} ticks (model-chosen)")
    log.info(f"  Max loss/day  : ${cfg['max_daily_loss']}")
    log.info(f"  Take profit   : ${cfg['take_profit']}")
    log.info(f"  Reselect every: {cfg['reselect_every']} live ticks")

    # -- Phase 1: Multi-Symbol Data Collection ----------------------------
    log.info("\n>> PHASE 1 -- Parallel Data Collection (all symbols)")
    symbol_data = MultiSymbolCollector(cfg).collect_all()
    log.info(f"  Collected data for {len(symbol_data)} symbols: "
             f"{list(symbol_data.keys())}")

    # -- Phase 2: Score all symbols, pick the best one --------------------
    log.info("\n>> PHASE 2 -- Symbol Scoring & Dual-Head LSTM Training")
    selector = SymbolSelector(cfg)
    best_sym, dual_lstm, fe, scores = selector.select(symbol_data)
    log.info(f"  [OK] Trading symbol selected: {best_sym}")

    # -- Phase 3: Live Trading --------------------------------------------
    log.info(f"\n>> PHASE 3 -- Live Adaptive Trading on {best_sym}")

    staker    = MartingaleStaker(
        cfg["base_stake"], cfg["martingale_mult"],
        cfg["martingale_after_loss"], cfg["max_martingale_steps"]
    )
    threshold = AdaptiveThreshold(cfg)
    trader    = LiveTrader(
        cfg, dual_lstm, fe, staker, threshold,
        active_symbol=best_sym,
        symbol_data=symbol_data,
    )
    trader.run()


if __name__ == "__main__":
    main()
