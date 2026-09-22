import sys
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import app_trim  # noqa: E402

y = np.zeros(int(16000 * 10), np.float32)
y[2 * 16000:5 * 16000] = (0.3 * np.sin(2 * np.pi * 300 * np.arange(int(16000 * 3)) / 16000)).astype(np.float32)
s, t = app_trim.trim_signal(y)
assert (s, s + len(t)) == (31232, 81408), (s, s + len(t))
n8 = 16000 * 8
assert len(app_trim.trim_signal(np.zeros(n8, np.float32))[1]) == n8
assert app_trim.reject_reasons(np.zeros(3200, np.float32)) == 'silent'
assert app_trim.reject_reasons(np.concatenate([np.zeros(3200), np.array([np.nan])]).astype(np.float32)) == 'non_finite'
ok, reason, x = app_trim.preprocess_to_embedding_input(np.full(5000, 0.2, np.float32))
assert ok and len(x) == 15360, (ok, len(x))
print('trim self-test OK (31232/81408, silent untrimmed, reject rules, 15360 pad)')