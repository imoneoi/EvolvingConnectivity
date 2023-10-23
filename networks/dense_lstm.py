import jax
import jax.numpy as jnp

import flax.linen as nn


class DenseLSTM(nn.Module):
    out_dims: int

    hidden_dims: int = 128

    @nn.compact
    def __call__(self, carry, input):
        new_carry, output = nn.OptimizedLSTMCell()(carry, input)
        output            = nn.Dense(self.out_dims)(output)

        return new_carry, output

    def initial_carry(self, key, batch_size):
        c = jnp.zeros((batch_size, self.hidden_dims))
        h = jnp.zeros((batch_size, self.hidden_dims))
        return c, h
