"""SlumberWell shared core: preprocessing, epoching, labelling and feature extraction.

Imported by BOTH the training notebook and the FastAPI backend so that training
and inference produce identical features (no training/serving skew).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import butter, resample_poly, sosfiltfilt, welch

# --------------------------------------------------------------------------- #
# Labels (spec section 4.4)
# --------------------------------------------------------------------------- #
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
LABEL_TO_INT = {name: i for i, name in enumerate(STAGE_NAMES)}
INT_TO_LABEL = {i: name for name, i in LABEL_TO_INT.items()}

# Hypnogram annotation text -> SlumberWell stage. Movement / unscored are absent
# on purpose: they are excluded, never treated as a sleep stage.
ANNOTATION_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",   # stages 3 + 4 merged (AASM convention)
    "Sleep stage R": "REM",
}

DEFAULT_CONFIG = {
    "channels": ["EEG Fpz-Cz"],
    "sampling_rate_hz": 100,
    "bandpass_hz": [0.3, 35.0],
    "filter_order": 4,
    "epoch_length_sec": 30,
    "welch": {"window_sec": 4, "overlap": 0.5},
    "bands_hz": {"delta": [0.5, 4.0], "theta": [4.0, 8.0],
                 "alpha": [8.0, 13.0], "beta": [13.0, 30.0]},
    "total_power_range_hz": [0.5, 30.0],
    "min_duration_epochs": 10,
    # training-only parameters (filled in by the notebook, stored for the record)
    "wake_margin_min": 30,
    "quality": {"flat_std_uv": 0.1, "max_abs_uv": None},
}


# --------------------------------------------------------------------------- #
# File-name helpers (spec section 4.2)
# --------------------------------------------------------------------------- #
def parse_sc_name(path) -> dict:
    """SC4ssNE0-PSG.edf -> subject number ss and night N."""
    name = Path(path).name
    m = re.match(r"SC4(\d{2})(\d)", name)
    if not m:
        raise ValueError(f"Not a Sleep Cassette file name: {name}")
    return {"subject": int(m.group(1)), "night": int(m.group(2))}


def find_hypnogram(psg_path) -> Path | None:
    """SC4001E0-PSG.edf -> SC4001E?-Hypnogram.edf in the same folder."""
    psg_path = Path(psg_path)
    prefix = psg_path.name.split("-")[0][:7]          # e.g. SC4001E
    hits = sorted(psg_path.parent.glob(f"{prefix}*-Hypnogram.edf"))
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# Signal loading and preprocessing (spec section 5, steps 1, 3, 4, 5)
# --------------------------------------------------------------------------- #
def read_edf(path, channels, verbose=False):
    """Read an EDF with MNE and return (signals_uV [n_ch, n_samples], sfreq, raw)."""
    import mne  # imported lazily so the module can be imported without MNE
    raw = mne.io.read_raw_edf(str(path), preload=False, verbose=verbose)
    missing = [c for c in channels if c not in raw.ch_names]
    if missing:
        raise KeyError(f"Channel(s) {missing} not found. Available channels: {raw.ch_names}")
    raw.pick(channels)
    raw.load_data(verbose=verbose)
    return raw.get_data() * 1e6, raw.info["sfreq"], raw      # volts -> microvolts


def resample_to(x, sfreq, target_sfreq):
    """Resample along the last axis to target_sfreq (no-op if already equal)."""
    if abs(sfreq - target_sfreq) < 1e-6:
        return x
    from fractions import Fraction
    frac = Fraction(target_sfreq / sfreq).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator, axis=-1)


def bandpass(x, fs, low, high, order=4):
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def make_epochs(x, fs, epoch_len_sec=30):
    """Cut [n_ch, n_samples] into non-overlapping epochs [n_epochs, n_ch, n_per_epoch].
    An incomplete final epoch is discarded (FR-03)."""
    n = int(round(epoch_len_sec * fs))
    n_epochs = x.shape[-1] // n
    x = x[..., : n_epochs * n]
    return x.reshape(x.shape[0], n_epochs, n).transpose(1, 0, 2)


def preprocess_signal(path, cfg):
    """EDF path -> epochs [n_epochs, n_ch, samples] in uV (resampled, band-passed)."""
    x, sfreq, raw = read_edf(path, cfg["channels"])
    fs = cfg["sampling_rate_hz"]
    x = resample_to(x, sfreq, fs)
    x = bandpass(x, fs, *cfg["bandpass_hz"], order=cfg.get("filter_order", 4))
    return make_epochs(x, fs, cfg["epoch_length_sec"]), raw


# --------------------------------------------------------------------------- #
# Labels from a hypnogram (spec section 5, step 6)
# --------------------------------------------------------------------------- #
def epoch_labels_from_annotations(onsets, durations, descriptions, n_epochs, epoch_len=30):
    """Return int array [n_epochs]; -1 = excluded (movement, unscored, uncovered).
    An epoch is labelled only if it lies entirely inside one mapped annotation."""
    labels = np.full(n_epochs, -1, dtype=int)
    eps = 1e-6
    for onset, dur, desc in zip(onsets, durations, descriptions):
        stage = ANNOTATION_MAP.get(str(desc).strip())
        if stage is None:
            continue
        first = int(np.ceil((onset - eps) / epoch_len))
        last = int(np.floor((onset + dur + eps) / epoch_len))   # exclusive
        first, last = max(first, 0), min(last, n_epochs)
        if last > first:
            labels[first:last] = LABEL_TO_INT[stage]
    return labels


def read_hypnogram(hyp_path, raw):
    """Read hypnogram annotations; return onsets relative to the PSG start."""
    import mne
    ann = mne.read_annotations(str(hyp_path))
    onsets = np.asarray(ann.onset, dtype=float)
    if ann.orig_time is not None and raw.info["meas_date"] is not None:
        onsets = onsets + (ann.orig_time - raw.info["meas_date"]).total_seconds()
    return onsets, np.asarray(ann.duration, dtype=float), list(ann.description)


def trim_wake_mask(labels, margin_epochs):
    """Keep wake only within margin_epochs before the first / after the last sleep epoch."""
    sleep_idx = np.where(labels > LABEL_TO_INT["W"])[0]
    keep = np.zeros(len(labels), dtype=bool)
    if len(sleep_idx) == 0:
        return keep
    lo = max(sleep_idx[0] - margin_epochs, 0)
    hi = min(sleep_idx[-1] + margin_epochs + 1, len(labels))
    keep[lo:hi] = True
    return keep


# --------------------------------------------------------------------------- #
# Feature engineering (spec section 6): 16 features per channel
# --------------------------------------------------------------------------- #
BASE_FEATURES = (
    ["mean", "std", "var", "rms", "min", "max", "range"]
    + ["delta_power", "theta_power", "alpha_power", "beta_power"]
    + ["rel_delta", "rel_theta", "rel_alpha", "rel_beta"]
    + ["spectral_entropy"]
)


def feature_names(cfg) -> list[str]:
    chs = cfg["channels"]
    if len(chs) == 1:
        return list(BASE_FEATURES)
    safe = [c.replace(" ", "_").replace("-", "_") for c in chs]
    return [f"{c}__{f}" for c in safe for f in BASE_FEATURES]


def _channel_features(ep, cfg):
    """ep: [n_epochs, samples] -> dict of feature arrays [n_epochs]."""
    fs = cfg["sampling_rate_hz"]
    f = {}
    f["mean"] = ep.mean(axis=1)
    f["std"] = ep.std(axis=1)
    f["var"] = ep.var(axis=1)
    f["rms"] = np.sqrt((ep ** 2).mean(axis=1))
    f["min"] = ep.min(axis=1)
    f["max"] = ep.max(axis=1)
    f["range"] = f["max"] - f["min"]

    nperseg = int(cfg["welch"]["window_sec"] * fs)
    noverlap = int(nperseg * cfg["welch"]["overlap"])
    freqs, psd = welch(ep, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1)

    def band_power(lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        return trapezoid(psd[:, m], freqs[m], axis=1)

    bands = {k: band_power(*v) for k, v in cfg["bands_hz"].items()}
    tlo, thi = cfg["total_power_range_hz"]
    total = band_power(tlo, thi)
    for k in ("delta", "theta", "alpha", "beta"):
        f[f"{k}_power"] = bands[k]
        f[f"rel_{k}"] = bands[k] / np.maximum(total, 1e-12)

    m = (freqs >= tlo) & (freqs <= thi)
    p = psd[:, m]
    p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    ent = -(p * np.log2(np.maximum(p, 1e-12))).sum(axis=1)
    f["spectral_entropy"] = ent / np.log2(p.shape[1])          # normalised to 0..1
    return f


def extract_features(epochs, cfg) -> pd.DataFrame:
    """epochs [n_epochs, n_ch, samples] -> DataFrame with columns in feature_names(cfg) order."""
    cols = {}
    names = feature_names(cfg)
    k = 0
    for ch in range(epochs.shape[1]):
        feats = _channel_features(epochs[:, ch, :], cfg)
        for base in BASE_FEATURES:
            cols[names[k]] = feats[base]
            k += 1
    return pd.DataFrame(cols, columns=names)


# --------------------------------------------------------------------------- #
# End-to-end helpers
# --------------------------------------------------------------------------- #
def featurize_recording(psg_path, hyp_path, cfg, subject_id=None, recording_id=None):
    """TRAINING path: PSG + hypnogram -> labelled feature table (before wake trimming)."""
    epochs, raw = preprocess_signal(psg_path, cfg)
    feats = extract_features(epochs, cfg)
    onsets, durs, descs = read_hypnogram(hyp_path, raw)
    labels = epoch_labels_from_annotations(onsets, durs, descs, len(feats), cfg["epoch_length_sec"])
    meta = pd.DataFrame({
        "subject_id": subject_id,
        "recording_id": recording_id or Path(psg_path).name.split("-")[0],
        "epoch_index": np.arange(len(feats)),
        "label": labels,
    })
    return pd.concat([meta, feats], axis=1)


def featurize_edf(path, cfg) -> pd.DataFrame:
    """INFERENCE path: EDF only -> feature matrix. Epochs start at time zero."""
    epochs, _ = preprocess_signal(path, cfg)
    if len(epochs) < cfg.get("min_duration_epochs", 1):
        raise ValueError(f"Recording too short: {len(epochs)} complete epochs.")
    return extract_features(epochs, cfg)


def summarize_predictions(stages, epoch_len_sec=30) -> dict:
    """Summary sleep metrics from predicted stage names (spec section 2.5)."""
    stages = np.asarray(stages)
    n = len(stages)
    mins = epoch_len_sec / 60.0
    trt = n * mins
    asleep = stages != "W"
    tst = asleep.sum() * mins
    onset = int(np.argmax(asleep)) if asleep.any() else None
    rem_idx = np.where(stages == "REM")[0]
    rem_lat = None
    if onset is not None and len(rem_idx):
        rem_lat = float((rem_idx[0] - onset) * mins)
    counts = {s: int((stages == s).sum()) for s in STAGE_NAMES}
    return {
        "total_recording_min": trt,
        "total_sleep_min": float(tst),
        "sleep_efficiency_pct": float(100 * tst / trt) if trt else 0.0,
        "sleep_onset_latency_min": None if onset is None else float(onset * mins),
        "rem_latency_min": rem_lat,
        "stage_counts": counts,
        "stage_percent": {s: float(100 * c / n) if n else 0.0 for s, c in counts.items()},
    }


# --------------------------------------------------------------------------- #
# Recording-level feature transforms (identical in training and serving)
# --------------------------------------------------------------------------- #
CONTEXT_BASE = ["std", "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "spectral_entropy"]
CONTEXT_SHIFTS = (1, 2)      # previous / next 1 and 2 epochs (30 s and 60 s on either side)


def _context_source_cols(cfg):
    return [n for n in feature_names(cfg) if n.split("__")[-1] in CONTEXT_BASE]


def model_feature_names(cfg) -> list[str]:
    """Columns the model sees: base features (+ neighbouring-epoch context if enabled)."""
    names = list(feature_names(cfg))
    if cfg.get("context_epochs"):
        for c in _context_source_cols(cfg):
            for k in CONTEXT_SHIFTS:
                names += [f"{c}_prev{k}", f"{c}_next{k}"]
    return names


def add_recording_features(feats: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """feats = ONE recording's epochs in time order (needs the feature_names(cfg) columns).

    1) cfg["normalize_per_recording"]: z-score every feature within the recording, which removes
       person-to-person amplitude differences (the main cross-subject problem for EEG features).
    2) cfg["context_epochs"]: add the previous / next epochs' values of a few key features,
       because sleep stages last minutes, not 30 s.
    """
    out = feats.copy()
    names = feature_names(cfg)
    if cfg.get("normalize_per_recording"):
        x = out[names].astype(float)
        out[names] = (x - x.mean()) / (x.std(ddof=0) + 1e-8)
    if cfg.get("context_epochs"):
        src = _context_source_cols(cfg)
        new = {}
        for k in CONTEXT_SHIFTS:
            prev = out[src].shift(k).fillna(out[src])       # edges: fall back to the epoch itself
            nxt = out[src].shift(-k).fillna(out[src])
            for c in src:
                new[f"{c}_prev{k}"] = prev[c]
                new[f"{c}_next{k}"] = nxt[c]
        out = pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)
    return out
