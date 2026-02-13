import jax
import jax.numpy as jnp
import equinox as eqx
from linax.models.ssm import SSM
from linax.models.linoss import LinOSSConfig
from linax.encoder import LinearEncoderConfig
from linax.heads import ClassificationHeadConfig
from linax.sequence_mixers.linoss import LinOSSSequenceMixerConfig

def count_parameters(model):
    """
    Count total number of trainable parameters in an Equinox model.
    """
    # Filter for arrays only (parameters)
    params = eqx.filter(model, eqx.is_array)
    
    # Count parameters in each array
    param_counts = jax.tree_util.tree_map(lambda x: x.size, params)
    
    # Sum all parameters
    total = sum(jax.tree_util.tree_leaves(param_counts))
    
    return total

def count_parameters_detailed(model):
    """
    Count parameters with more detail, showing breakdown by component.
    """
    params = eqx.filter(model, eqx.is_array)
    
    total = 0
    print("\nParameter breakdown:")
    print("-" * 60)
    
    # Flatten the tree and get paths
    flat_params = jax.tree_util.tree_leaves_with_path(params)
    
    for path, param in flat_params:
        if param is not None and hasattr(param, 'shape'):
            param_count = param.size
            total += param_count
            
            # Create readable path name
            path_str = '.'.join(str(k.key) for k in path if hasattr(k, 'key'))
            print(f"{path_str:50s} {str(param.shape):20s} {param_count:>10,}")
    
    print("-" * 60)
    print(f"{'TOTAL':50s} {total:>31,}")
    print("-" * 60)
    
    return total


# Create the model
num_classes = 9
key = jax.random.PRNGKey(0)

cfg = LinOSSConfig(
    encoder_config=LinearEncoderConfig(in_features=1, out_features=64),
    head_config=ClassificationHeadConfig(out_features=num_classes, reduce=True),
    num_blocks=2,
    sequence_mixer_config=LinOSSSequenceMixerConfig(state_dim=64),
)

model = SSM(cfg, key)

# Count parameters
print("\n" + "="*60)
print("MODEL PARAMETER COUNT")
print("="*60)

# Simple count
total_params = count_parameters(model)
print(f"\nTotal parameters: {total_params:,}")

# Detailed breakdown
total_params_detailed = count_parameters_detailed(model)

# Verify they match
assert total_params == total_params_detailed, "Counts don't match!"

print(f"\n✓ Model has {total_params:,} trainable parameters")