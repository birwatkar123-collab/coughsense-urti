"""NumPy port of the research V2/V4 CPU preprocessing, up to and including the mean
YAMNet embedding frame-count arithmetic. This is the reference implementation the
Android pipeline (Kotlin) must mirror exactly.

Matches librosa.effects.trim(y, top_db=30) / librosa.feature.rms defaults used by the
research pipeline (frame_length=2048, hop_length=512, center=True pad 1024) and the
research accept/reject rules from shared-training-flat.json:

    SR=16000, max 60s; reject len<3200, non-finite, or max|y|<1e-5 (silent);
    trim(top_db=30); reject trimmed<3200; clip to [-1,1]; pad to min 0.96s (15360).

All arithmetic mirrors librosa's float32 RMS power path so that Kotlin (Float32) can
reproduce identical trim boundaries.
"""
import numpy as np

SR = 16000
MIN_SAMPLES = 3200  # 0.2 s
PAD_SAMPLES = 15360  # 0.96 s min after clipping
TRIM_DB = 30.0
FRAME_LENGTH = 2048
HOP_LENGTH = 512
CENTER = FRAME_LENGTH // 2  # 1024
SILENT_THRESHOLD = 1e-5


def trim_signal(y: np.ndarray) -> tuple[int, np.ndarray]:
    """Return (start_sample, trimmed float32 array) mirroring librosa.effects.trim(y, top_db=30)."""
    x = np.asarray(y, dtype=np.float32)
    n = x.shape[0]
    if n == 0:
        return 0, x
    padded = np.pad(x, (CENTER, CENTER), mode='constant')
    n_frames = max((padded.shape[0] - FRAME_LENGTH) // HOP_LENGTH + 1, 0)
    if n_frames == 0:
        return 0, x
    idx = (np.arange(FRAME_LENGTH)[:, None] * np.int64(1) +
           np.arange(n_frames)[None, :] * HOP_LENGTH)
    frames = padded[idx]  # (frame_length, n_frames) float32 view
    # Sequential float32 accumulation (identical to the Kotlin port; equal-frame
    # bookkeeping as librosa.effects.trim via feature.rms defaults, same boundaries).
    power = np.empty(n_frames, dtype=np.float32)
    for f in range(n_frames):
        acc = np.float32(0.0)
        for s in range(FRAME_LENGTH):
            v = frames[s, f]
            acc = np.float32(acc + np.float32(v * v))
        power[f] = np.float32(acc / np.float32(FRAME_LENGTH))
    rms = np.sqrt(power).astype(np.float32)
    # amplitude_to_db(rms, ref=np.max, top_db=None), librosa power_to_db amin=1e-10
    rms_sq = (rms * rms).astype(np.float32)
    amin = np.float32(1e-10)
    amin2 = np.float32(amin * amin)  # 1e-20
    ref = np.float32(rms_sq.max())
    log_min = np.float32(10.0 * np.log10(np.maximum(amin2, ref)))
    db = (np.float32(10.0) * np.log10(np.maximum(amin2, rms_sq)) - log_min).astype(np.float32)
    non_silent = np.flatnonzero(db > np.float32(-TRIM_DB))
    if non_silent.size == 0:
        return 0, x
    start = int(non_silent[0]) * HOP_LENGTH
    end = min(n, (int(non_silent[-1]) + 1) * HOP_LENGTH)
    return start, x[start:end].copy()


def reject_reasons(samples: np.ndarray) -> str:
    """Return reason string if the raw decoded audio is rejected, else ''."""
    if samples.shape[0] < MIN_SAMPLES:
        return 'too_short'
    if not np.all(np.isfinite(samples)):
        return 'non_finite'
    if float(np.max(np.abs(samples))) < SILENT_THRESHOLD:
        return 'silent'
    return ''


def preprocess_to_embedding_input(samples: np.ndarray) -> tuple[bool, str, np.ndarray]:
    """Research preprocessing for one decoded float32 mono-16k signal.

    Returns (ok, reason, processed float32 waveform ready for YAMNet).
    Mirrors shared-training-flat.json; applied identically by the Android app.
    """
    reason = reject_reasons(samples)
    if reason:
        return False, reason, np.empty(0, np.float32)
    start, trimmed = trim_signal(samples)
    if trimmed.shape[0] < MIN_SAMPLES:
        return False, 'trimmed_too_short'
    clipped = np.clip(trimmed, -1.0, 1.0).astype(np.float32)
    if clipped.shape[0] < PAD_SAMPLES:
        clipped = np.pad(clipped, (0, PAD_SAMPLES - clipped.shape[0]), mode='constant')
    return True, '', clipped


def frame_count_for_length(samples: int) -> int:
    """librosa.feature.rms centered frame count for a sample length (same formula as YAMNet
    'sort_samples' frame dimension)."""
    padded = samples + 2 * CENTER
    return max((padded - FRAME_LENGTH) // HOP_LENGTH + 1, 0)