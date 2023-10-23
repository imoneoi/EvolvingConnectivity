from typing import Any, Tuple
import time

import jax
import jax.numpy as jnp
import flax.linen as nn

from conn_snn import ConnSNN


jax.config.update("jax_default_prng_impl", "rbg")


def rand_bernoulli_like(key: Any, params: Any, p: float, batch_size: Tuple = ()) -> Any:
    num_vars = len(jax.tree_util.tree_leaves(params))
    treedef = jax.tree_util.tree_structure(params)

    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_util.tree_map(
        lambda x, k: jax.random.uniform(k, (*batch_size, *x.shape)) < p,
        params, jax.tree_util.tree_unflatten(treedef, all_keys))

    return noise


def test_snn_speed(
    seed:        int = 0,
    
    num_neurons: int = 256,

    batch_size: int = 10240,
    steps:      int = 1000,

    in_dims:    int = 240,
    out_dims:   int = 17,

    p: float = 0.5,

    neuron_dtype: jnp.dtype = jnp.float32,
    spike_dtype:  jnp.dtype = jnp.int8
):
    key, key_net, key_carry = jax.random.split(jax.random.PRNGKey(seed), 3)

    # Create network class
    network = nn.vmap(ConnSNN,
                      variable_axes={"params": 0},
                      split_rngs={"params": True})
    network = network(out_dims=out_dims,
                      num_neurons=num_neurons,
                      neuron_dtype=neuron_dtype,
                      spike_dtype=spike_dtype)

    # Initialize network
    carry = network.initial_carry(key_carry, batch_size)
    input_example = jnp.zeros((batch_size, in_dims), neuron_dtype)

    params = jax.jit(network.init)(key_net, carry, input_example)
    params = jax.jit(rand_bernoulli_like)(key_net, params, p)

    # Forward network
    forward_jit = jax.jit(network.apply, donate_argnums=(1,))
    metrics_jit = jax.jit(network.carry_metrics)
    # Warmup JIT
    carry, output = forward_jit(params, carry, input_example)

    start_time = time.time()
    for step in range(steps):
        # generate inputs
        key, key_inputs = jax.random.split(key)
        inputs = jax.random.normal(key_inputs, (batch_size, in_dims), neuron_dtype)
        # forward
        carry, output = forward_jit(params, carry, inputs)
        metrics = metrics_jit(carry)

        print(f"{jnp.mean(output)},{jnp.mean(jnp.std(output, axis=-1))},{metrics}")

    elapsed = time.time() - start_time
    print(f"Elapsed: {elapsed:.2f}")


if __name__ == "__main__":
    test_snn_speed()
