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
    # Load with INTERPOLATION (best for SSMs)
    # Options: "interpolate", "forward_fill", "mean", "zero"
    X, y = load_dataset("light_data/keplerq9v3", 
                        normalize=True, 
                        nan_strategy="interpolate")
    
    # Diagnostic checks
    print("X finite:", bool(jnp.isfinite(X).all()))
    print("y finite:", bool(jnp.isfinite(y).all()))
    print("X min/max:", float(jnp.nanmin(X)), float(jnp.nanmax(X)))
    print("Any NaNs in X:", bool(jnp.isnan(X).any()))
    print("Any infs in X:", bool(jnp.isinf(X).any()))
    
    # Final safety check
    if jnp.isnan(X).any() or jnp.isinf(X).any():
        print("⚠️  WARNING: Still have NaN/inf after loading, applying emergency cleanup")
        X = jnp.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)
    
    # Use shorter training windows
    seq_len = 512
    T = X.shape[1]
    if T <= seq_len:
        raise ValueError(f"Sequence too short: T={T}, need > {seq_len}")
    
    # Crop a fixed window
    X = X[:, :seq_len]               # (N, L)
    X = jnp.expand_dims(X, -1)       # (N, L, 1)
    
    # OPTIONAL: Create attention mask to distinguish real data from padding
    # This tells the model which timesteps are real vs padded
    # For now, we'll assume all data within seq_len is real
    
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
    
    # More aggressive partitioning - only keep float/complex arrays
    def is_inexact_array(x):
        """Only return True for float or complex arrays, not bool/int"""
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
        # Reconstruct model
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
        
        # Light NaN handling (should rarely trigger with good preprocessing)
        logits = jnp.where(jnp.isfinite(logits), logits, 0.0)
        logits = jnp.clip(logits, -50.0, 50.0)
        
        st2 = jax.tree_util.tree_map(lambda z: z[0], st2s)
        
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()
        acc = (jnp.argmax(logits, axis=-1) == yb).mean()
        
        return loss, (st2, acc)
    
    @eqx.filter_jit
    def train_step(params_tree, static_tree, st, opt_st, xb, yb, k):
        # Create closure that only exposes params_tree for differentiation
        def loss_fn(p):
            return loss_and_state(p, static_tree, st, xb, yb, k)
        
        # Take gradient only with respect to params
        (loss, (st2, acc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_tree)
        
        # Clean gradients (should be rare with good data)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, 0.0),
            grads
        )
        
        updates, opt_st2 = opt.update(grads, opt_st)
        params_tree2 = optax.apply_updates(params_tree, updates)
        
        return params_tree2, st2, opt_st2, loss, acc
    
    epochs = 100
    batch_size = 16
    
    for ep in range(1, epochs + 1):
        key, k_epoch = jax.random.split(key)
        total_loss = 0.0
        total_acc = 0.0
        steps = 0
        
        for xb, yb, _ in make_batches(X, y, batch_size, k_epoch):
            key, k_step = jax.random.split(key)
            params, state, opt_state, loss, acc = train_step(
                params, static, state, opt_state, xb, yb, k_step
            )
            
            total_loss += float(loss)
            total_acc += float(acc)
            steps += 1
        
        print(f"Epoch {ep}: loss={total_loss/steps:.4f}  acc={total_acc/steps:.4f}")
    
    # Save
    final_model = eqx.combine(params, static)
    eqx.tree_serialise_leaves("linoss_classifier.eqx", eqx.filter(final_model, eqx.is_array))
    print("Saved -> linoss_classifier.eqx")


if __name__ == "__main__":
    main()