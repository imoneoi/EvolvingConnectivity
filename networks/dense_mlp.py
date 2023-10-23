import jax
import jax.numpy as jnp

import flax.linen as nn


class DenseMLP(nn.Module):
    out_dims: int

    hidden_dims: int = 256

    @nn.compact
    def __call__(self, carry, input):
        hidden = jax.nn.tanh(nn.Dense(self.hidden_dims)(input))
        output = nn.Dense(self.out_dims)(hidden)

        return None, output

    def initial_carry(self, key, batch_size):
        return None
