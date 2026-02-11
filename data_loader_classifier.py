import os
import glob
import numpy as np
import jax.numpy as jnp

CLASS_NAMES = [
    "APERIODIC",
    "CONSTANT",
    "CONTACT_ROT",
    "DSCT_BCEP",
    "ECLIPSE",
    "GDOR_SPB",
    "INSTRUMENT",
    "RRLYR_CEPHEID",
    "SOLARLIKE",
]

def load_dataset(root="light_data/keplerq9v3", normalize=True):
    xs = []
    ys = []

    for class_id, cname in enumerate(CLASS_NAMES):
        folder = os.path.join(root, cname)
        files = sorted(glob.glob(os.path.join(folder, "*.txt")))

        for f in files:
            arr = np.loadtxt(f)
            time, flux, base = arr.T

            if normalize:
                flux = (flux - flux.mean()) / flux.std()  # original behavior

            xs.append(flux.astype(np.float32))
            ys.append(class_id)

    # pad to same length
    T = max(len(x) for x in xs)
    xpad = np.zeros((len(xs), T), dtype=np.float32)
    for i, x in enumerate(xs):
        xpad[i, :len(x)] = x

    X = jnp.asarray(xpad)           # (N, T)
    y = jnp.asarray(np.array(ys))   # (N,)
    return X, y

