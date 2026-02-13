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

def load_dataset(root="light_data/keplerq9v3", normalize=True, nan_strategy="interpolate", 
                 split_ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Load light curve dataset with SSM-aware NaN handling.
    
    Args:
        root: Path to data directory
        normalize: Whether to normalize flux values
        nan_strategy: How to handle NaNs
            - "interpolate": Linear interpolation (best for SSMs)
            - "forward_fill": Carry forward last valid value
            - "zero": Replace with 0 (NOT recommended for SSMs)
            - "mean": Replace with local mean
        split_ratios: (train, val, test) ratios. Set to None for no split.
        seed: Random seed for reproducible splits
    
    Returns:
        If split_ratios is None: (X, y)
        If split_ratios is tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    xs = []
    ys = []
    nan_stats = {"total_nans": 0, "files_with_nans": 0, "total_points": 0}
    
    for class_id, cname in enumerate(CLASS_NAMES):
        folder = os.path.join(root, cname)
        files = sorted(glob.glob(os.path.join(folder, "*.txt")))
        
        for f in files:
            arr = np.loadtxt(f)
            time, flux, base = arr.T
            
            # Track NaN statistics
            nan_mask = np.isnan(flux) | np.isinf(flux)
            num_nans = nan_mask.sum()
            nan_stats["total_nans"] += num_nans
            nan_stats["total_points"] += len(flux)
            if num_nans > 0:
                nan_stats["files_with_nans"] += 1
            
            # Handle NaNs based on strategy
            if nan_strategy == "interpolate":
                flux = _interpolate_nans(flux)
            elif nan_strategy == "forward_fill":
                flux = _forward_fill_nans(flux)
            elif nan_strategy == "mean":
                flux = _mean_fill_nans(flux)
            elif nan_strategy == "zero":
                flux = np.nan_to_num(flux, nan=0.0, posinf=1e10, neginf=-1e10)
            else:
                raise ValueError(f"Unknown nan_strategy: {nan_strategy}")
            
            # Handle any remaining infs
            flux = np.clip(flux, -1e10, 1e10)
            
            if normalize:
                mean = np.mean(flux)
                std = np.std(flux)
                if std > 1e-8:  # Avoid division by very small numbers
                    flux = (flux - mean) / std
                else:
                    flux = flux - mean
                    
            xs.append(flux.astype(np.float32))
            ys.append(class_id)
    
    # Print NaN statistics
    if nan_stats["total_nans"] > 0:
        pct = 100 * nan_stats["total_nans"] / nan_stats["total_points"]
        print(f"\n⚠️  NaN Statistics:")
        print(f"  - Files with NaNs: {nan_stats['files_with_nans']}")
        print(f"  - Total NaN points: {nan_stats['total_nans']} ({pct:.2f}%)")
        print(f"  - Strategy used: {nan_strategy}\n")
    
    # Pad to same length
    T = max(len(x) for x in xs)
    xpad = np.zeros((len(xs), T), dtype=np.float32)
    for i, x in enumerate(xs):
        xpad[i, :len(x)] = x
    
    X = jnp.asarray(xpad)           # (N, T)
    y = jnp.asarray(np.array(ys))   # (N,)
    
    # Split data if requested
    if split_ratios is None:
        return X, y
    
    # Validate split ratios
    assert len(split_ratios) == 3, "split_ratios must be (train, val, test)"
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to 1.0"
    
    # Stratified split to maintain class balance
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    # Shuffle with seed for reproducibility
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    
    # Calculate split points
    train_ratio, val_ratio, test_ratio = split_ratios
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # Split indices
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    # Create splits
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"\n📊 Dataset Split:")
    print(f"  - Train: {len(train_idx)} samples ({100*train_ratio:.1f}%)")
    print(f"  - Val:   {len(val_idx)} samples ({100*val_ratio:.1f}%)")
    print(f"  - Test:  {len(test_idx)} samples ({100*test_ratio:.1f}%)")
    
    # Print class distribution
    print(f"\n📈 Class Distribution (Train set):")
    for class_id, cname in enumerate(CLASS_NAMES):
        count = (y_train == class_id).sum()
        print(f"  - {cname:20s}: {count:5d} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _interpolate_nans(flux):
    """
    Linear interpolation for NaN values.
    Best for SSMs as it maintains temporal continuity.
    """
    flux = flux.copy()
    
    # Handle infs first
    flux[np.isinf(flux)] = np.nan
    
    nan_mask = np.isnan(flux)
    if not nan_mask.any():
        return flux
    
    # Get valid indices
    valid_idx = np.where(~nan_mask)[0]
    
    if len(valid_idx) == 0:
        # All NaN - replace with zeros
        return np.zeros_like(flux)
    
    if len(valid_idx) == 1:
        # Only one valid point - use it everywhere
        return np.full_like(flux, flux[valid_idx[0]])
    
    # Interpolate
    flux[nan_mask] = np.interp(
        np.where(nan_mask)[0],  # x-coordinates to interpolate
        valid_idx,               # x-coordinates of data
        flux[valid_idx]          # y-coordinates of data
    )
    
    return flux


def _forward_fill_nans(flux):
    """
    Forward fill: carry last valid observation forward.
    Good for SSMs when you expect values to change slowly.
    """
    flux = flux.copy()
    flux[np.isinf(flux)] = np.nan
    
    nan_mask = np.isnan(flux)
    if not nan_mask.any():
        return flux
    
    # Find first valid value
    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        return np.zeros_like(flux)
    
    # Fill leading NaNs with first valid value
    first_valid = valid_idx[0]
    flux[:first_valid] = flux[first_valid]
    
    # Forward fill
    last_valid = flux[first_valid]
    for i in range(first_valid + 1, len(flux)):
        if np.isnan(flux[i]):
            flux[i] = last_valid
        else:
            last_valid = flux[i]
    
    return flux


def _mean_fill_nans(flux, window=10):
    """
    Replace NaNs with local windowed mean.
    Reasonable for SSMs but can introduce discontinuities.
    """
    flux = flux.copy()
    flux[np.isinf(flux)] = np.nan
    
    nan_mask = np.isnan(flux)
    if not nan_mask.any():
        return flux
    
    for i in np.where(nan_mask)[0]:
        # Get window around the NaN
        start = max(0, i - window)
        end = min(len(flux), i + window + 1)
        window_vals = flux[start:end]
        
        # Compute mean of valid values in window
        valid_vals = window_vals[~np.isnan(window_vals)]
        if len(valid_vals) > 0:
            flux[i] = np.mean(valid_vals)
        else:
            flux[i] = 0.0  # Fallback if entire window is NaN
    
    return flux