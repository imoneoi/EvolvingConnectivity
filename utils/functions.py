from functools import partial
from typing import Any, Dict, Tuple
import math
import os
import pickle

import jax
import jax.numpy as jnp

import flax


Params = flax.core.FrozenDict[str, Any]


# ---------- Numpy ----------


@partial(jax.jit, static_argnames=["axis"])
def finitemean(x, axis=None):
    """Mean of x ignoring NaN and Inf"""
    mask = jnp.isfinite(x)
    return jnp.sum(jnp.where(mask, x, 0), axis=axis) / jnp.sum(mask, axis=axis)


# ---------- Pytree ----------


def zeros_like_tree(tree: Any, batch_shape: Tuple = ()):
    """Returns a pytree containing zeros with identical shape as `tree`"""
    return jax.tree_map(lambda x: jnp.zeros_like(x, shape=(*batch_shape, *x.shape)), tree)


def rand_normal_like_tree(key: Any, params: Params, std: float = 1.0, batch_shape: Tuple = ()):
    """Return a pytree like `params` where every element follows standard normal distribution
       May add a batch dim on parameters with batch_shape=(bs,)
    """
    num_vars = len(jax.tree_util.tree_leaves(params))
    treedef = jax.tree_util.tree_structure(params)

    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_util.tree_map(
        lambda g, k: std * jax.random.normal(k, shape=(*batch_shape, *g.shape), dtype=g.dtype),
        params, jax.tree_util.tree_unflatten(treedef, all_keys))

    return noise


# ---------- Brax ----------


def shuffle_env_state(key: Any, states: Params):
    """Shuffles a batched Brax environment states along axis 0 (batch axis)
    """
    num_envs = jax.tree_util.tree_leaves(states)[0].shape[0]
    indices  = jax.random.randint(key, (num_envs, ), 0, num_envs)

    return jax.tree_util.tree_map(
        lambda x: x[indices], states
    )


# ---------- Metrics ----------


def mean_weight_abs(params: Params):
    nonzero = sum(jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(params))
    total   = sum(math.prod(x.shape)  for x in jax.tree_util.tree_leaves(params))
    return nonzero / total


def param_norm(params: Params):
    """Return L2 norm of flattened param vector
    """
    sumsq = [jnp.sum(leaf ** 2) for leaf in jax.tree_util.tree_leaves(params)]

    return jnp.sqrt(sum(sumsq))


# ---------- Serialization ----------


def save_obj_to_file(fn: str, obj: Any):
    os.makedirs(os.path.dirname(fn), exist_ok=True)

    with open(fn, "wb") as f:
        pickle.dump(obj, f)


def load_obj_from_file(fn: str):
    with open(fn, "rb") as f:
        obj = pickle.load(f)
    return obj
