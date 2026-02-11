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


def make_batches(X, y, batch_size, key):
    n = X.shape[0]
    perm = jax.random.permutation(key, n)
    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        yield X[idx], y[idx]


def main():
    # Load
    X, y = load_dataset("light_data/keplerq9v3", normalize=True)
    import jax.numpy as jnp

    print("X finite:", bool(jnp.isfinite(X).all()))
    print("y finite:", bool(jnp.isfinite(y).all()))
    print("X min/max:", float(jnp.min(X)), float(jnp.max(X)))
    print("Any NaNs in X:", bool(jnp.isnan(X).any()))
    print("Any infs in X:", bool(jnp.isinf(X).any()))

    # Sanitize inputs: replace NaNs/Infs so training doesn't fail.
    X = jnp.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)

    # Use shorter training windows (you can tune)
    seq_len = 512
    T = X.shape[1]
    if T <= seq_len:
        raise ValueError(f"Sequence too short: T={T}, need > {seq_len}")

    # Crop a fixed window for now (simple baseline)
    X = X[:, :seq_len]               # (N, L)
    X = jnp.expand_dims(X, -1)       # (N, L, 1)

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

    # IMPORTANT: create the Equinox state for this model
    state = eqx.nn.State(model)

    opt = optax.chain(
          optax.clip_by_global_norm(1.0),
          optax.adam(1e-4),
    )

    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    def _broadcast_state(st,B):
        return jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (B,) + a.shape), st)

    def loss_and_state(m, st, xb, yb, k):
    # xb: (B, L, 1)
        bsz = xb.shape[0]
        keys = jax.random.split(k, bsz)
        st_batched = _broadcast_state(st, bsz)
        def forward_one(x,st_i, kk):
           logits, st_o = m(x, st_i, kk)   # x: (L, 1)
           logits = jnp.real(logits)
           return logits, st_o

        logits, st2s = jax.vmap(forward_one, axis_name = "batch")(xb, st_batched, keys)  # (B, C)
        logits = jnp.real(logits)
        logits = jnp.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        logits = jnp.clip(logits, -50.0, 50.0)

        st2 = jax.tree_util.tree_map(lambda z: z[0], st2s)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()
        acc = (jnp.argmax(logits, axis=-1) == yb).mean()
        return loss, (st2, acc)
    @eqx.filter_jit
    def train_step(m, st, opt_st, xb, yb, k):
        (loss, (st2,acc)), grads = eqx.filter_value_and_grad(loss_and_state, has_aux=True)(
            m, st, xb, yb, k
        )
        # Do not pass `params=` when running inside a JITted function;
        # passing traced parameter trees (JitTracer) can cause tree-casting
        # mismatches with the optimizer state. Omitting `params` here is
        # fine because we are not using transforms that require params.
        updates, opt_st2 = opt.update(grads, opt_st)
        m2 = eqx.apply_updates(m, updates)
        return m2, st, opt_st2, loss, acc   # keep state unchanged

    epochs = 100
    batch_size = 16

    for ep in range(1, epochs + 1):
        key, k_epoch = jax.random.split(key)
        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        for xb, yb in make_batches(X, y, batch_size, k_epoch):
            key, k_step = jax.random.split(key)
            model, state, opt_state, loss, acc = train_step(model, state, opt_state, xb, yb, k_step)
            total_loss += float(loss)
            total_acc += float(acc)
            steps += 1

        print(f"Epoch {ep}: loss={total_loss/steps:.4f}  acc={total_acc/steps:.4f}")

    # Save params (arrays only)
    eqx.tree_serialise_leaves("linoss_classifier.eqx", eqx.filter(model, eqx.is_array))
    print("Saved -> linoss_classifier.eqx")


if __name__ == "__main__":
    main()

