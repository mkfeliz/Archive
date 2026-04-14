import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from linax.models.ssm import SSM
from linax.models.linoss import LinOSSConfig
from linax.encoder import LinearEncoderConfig
from linax.heads import ClassificationHeadConfig
from linax.sequence_mixers.linoss import LinOSSSequenceMixerConfig
from data_loader_classifier import load_dataset, CLASS_NAMES
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path


def make_batches(X, y, batch_size, key, mask=None):
    n = X.shape[0]
    perm = jax.random.permutation(key, n)
    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        if mask is not None:
            yield X[idx], y[idx], mask[idx]
        else:
            yield X[idx], y[idx], None


def main():
    # Load with proper train/val/test split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(
        "light_data/keplerq9v3",
        normalize=True,
        nan_strategy="interpolate",
        split_ratios=(0.7, 0.15, 0.15),
        seed=42
    )

    # Diagnostic checks on training set
    print("X_train finite:", bool(jnp.isfinite(X_train).all()))
    print("y_train finite:", bool(jnp.isfinite(y_train).all()))
    print("X_train min/max:", float(jnp.nanmin(X_train)), float(jnp.nanmax(X_train)))
    print("Any NaNs in X_train:", bool(jnp.isnan(X_train).any()))
    print("Any infs in X_train:", bool(jnp.isinf(X_train).any()))

    # Final safety check
    if jnp.isnan(X_train).any() or jnp.isinf(X_train).any():
        print("WARNING: Still have NaN/inf after loading, applying emergency cleanup")
        X_train = jnp.nan_to_num(X_train, nan=0.0, posinf=1e4, neginf=-1e4)
        X_val = jnp.nan_to_num(X_val, nan=0.0, posinf=1e4, neginf=-1e4)
        X_test = jnp.nan_to_num(X_test, nan=0.0, posinf=1e4, neginf=-1e4)

    # Use shorter training windows
    seq_len = 512
    T = X_train.shape[1]
    if T <= seq_len:
        raise ValueError(f"Sequence too short: T={T}, need > {seq_len}")

    # Crop all sets to same window
    X_train = X_train[:, :seq_len]
    X_val = X_val[:, :seq_len]
    X_test = X_test[:, :seq_len]

    # Add channel dimension
    X_train = jnp.expand_dims(X_train, -1)
    X_val = jnp.expand_dims(X_val, -1)
    X_test = jnp.expand_dims(X_test, -1)

    num_classes = len(CLASS_NAMES)
    key = jax.random.PRNGKey(0)
    key, k_model = jax.random.split(key)

    cfg = LinOSSConfig(
        encoder_config=LinearEncoderConfig(in_features=1, out_features=64),
        head_config=ClassificationHeadConfig(out_features=num_classes, reduce=True),
        num_blocks=2,
        sequence_mixer_config=LinOSSSequenceMixerConfig(state_dim=64),
    )

    model = SSM(cfg, k_model)
    state = eqx.nn.State(model)

    def is_inexact_array(x):
        return eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.inexact)

    params, static = eqx.partition(model, is_inexact_array)

    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-4),
    )
    opt_state = opt.init(params)

    def _broadcast_state(st, B):
        return jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (B,) + a.shape), st)

    def loss_and_state(params_tree, static_tree, st, xb, yb, k):
        m = eqx.combine(params_tree, static_tree)

        bsz = xb.shape[0]
        keys = jax.random.split(k, bsz)
        st_batched = _broadcast_state(st, bsz)

        def forward_one(x, st_i, kk):
            logits, st_o = m(x, st_i, kk)
            logits = jnp.real(logits)
            return logits, st_o

        logits, st2s = jax.vmap(forward_one, axis_name="batch")(xb, st_batched, keys)
        logits = jnp.real(logits)

        logits = jnp.where(jnp.isfinite(logits), logits, 0.0)
        logits = jnp.clip(logits, -50.0, 50.0)

        st2 = jax.tree_util.tree_map(lambda z: z[0], st2s)

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()
        acc = (jnp.argmax(logits, axis=-1) == yb).mean()

        return loss, (st2, acc)

    @eqx.filter_jit
    def train_step(params_tree, static_tree, st, opt_st, xb, yb, k):
        def loss_fn(p):
            return loss_and_state(p, static_tree, st, xb, yb, k)

        (loss, (st2, acc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_tree)

        grads = jax.tree_util.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)

        updates, opt_st2 = opt.update(grads, opt_st)
        params_tree2 = optax.apply_updates(params_tree, updates)

        return params_tree2, st2, opt_st2, loss, acc

    @eqx.filter_jit
    def eval_step(params_tree, static_tree, st, xb, yb, k):
        loss, (st2, acc) = loss_and_state(params_tree, static_tree, st, xb, yb, k)
        return loss, acc

    epochs = 150 # change this 
    batch_size = 32 # change this 

    best_val_acc = 0.0
    best_params = params

    print("\nStarting Training...")

    for ep in range(1, epochs + 1):
        key, k_epoch = jax.random.split(key)

        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        for xb, yb, _ in make_batches(X_train, y_train, batch_size, k_epoch):
            key, k_step = jax.random.split(key)
            params, state, opt_state, loss, acc = train_step(
                params, static, state, opt_state, xb, yb, k_step
            )

            total_loss += float(loss)
            total_acc += float(acc)
            steps += 1

        train_loss = total_loss / steps
        train_acc = total_acc / steps

        val_loss = 0.0
        val_acc = 0.0
        val_steps = 0

        key, k_val = jax.random.split(key)
        for xb, yb, _ in make_batches(X_val, y_val, batch_size, k_val):
            key, k_step = jax.random.split(key)
            loss, acc = eval_step(params, static, state, xb, yb, k_step)
            val_loss += float(loss)
            val_acc += float(acc)
            val_steps += 1

        val_loss /= val_steps
        val_acc /= val_steps

        print(
            f"Epoch {ep:3d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # Track best model only (no early stopping)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            print(f"  New best validation accuracy: {val_acc:.4f}")

    # Test set evaluation with best model
    print("\nEvaluating on Test Set...")
    params = best_params

    test_loss = 0.0
    test_acc = 0.0
    test_steps = 0
    all_predictions = []
    all_labels = []

    key, k_test = jax.random.split(key)
    for xb, yb, _ in make_batches(X_test, y_test, batch_size, k_test):
        key, k_step = jax.random.split(key)
        loss, acc = eval_step(params, static, state, xb, yb, k_step)
        test_loss += float(loss)
        test_acc += float(acc)
        test_steps += 1
        
        # Collect predictions and labels for confusion matrix
        model_combined = eqx.combine(params, static)
        bsz = xb.shape[0]
        keys = jax.random.split(k_step, bsz)
        st_batched = _broadcast_state(state, bsz)
        
        def forward_one(x, st_i, kk):
            logits, _ = model_combined(x, st_i, kk)
            return jnp.real(logits)
        
        logits = jax.vmap(forward_one, axis_name="batch")(xb, st_batched, keys)
        predictions = jnp.argmax(logits, axis=-1)
        
        all_predictions.extend(predictions.tolist())
        all_labels.extend(yb.tolist())

    test_loss /= test_steps
    test_acc /= test_steps

    print("\nFinal Test Results:")
    print(f"  - Test Loss: {test_loss:.4f}")
    print(f"  - Test Accuracy: {test_acc:.4f}")
    print(f"  - Best Val Accuracy: {best_val_acc:.4f}")

    # Calculate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(len(CLASS_NAMES))))
    
    # Create plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    save_dir = Path('.')
    save_path = save_dir / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

    final_model = eqx.combine(params, static)
    eqx.tree_serialise_leaves("linoss_classifier.eqx", eqx.filter(final_model, eqx.is_array))
    print("Saved -> linoss_classifier.eqx")


if __name__ == "__main__":
    main()
