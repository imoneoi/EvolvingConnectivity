import math

import jax
import jax.numpy as jnp
import flax.linen as nn


def lerp(y, x, alpha):
    # Linear interpolation
    # = alpha * y + (1 - alpha) * x
    return x + alpha * (y - x)


def conn_dense(kernel, x):
    # matmul
    return jax.lax.dot_general(x, kernel, (((x.ndim - 1,), (0,)), ((), ())))


class DenseSNN(nn.Module):
    """Recurrent spiking neural network with LIF model

    Same architecture and parameters as conn_snn, except using real weights."""

    # Network parameters
    out_dims: int

    num_neurons: int = 256

    rand_init_Vm: bool = True

    dtype: jnp.dtype = jnp.float32

    # SNN simulation
    sim_time: float = 16.6  # ms
    dt: float       = 0.5   # ms

    # SNN parameters
    K_in:  float = 0.1
    K_h:   float = 1.0
    K_out: float = 5.0

    tau_syn:  float = 5.0   # ms
    tau_Vm:   float = 10.0  # ms
    tau_out:  float = 10.0  # ms

    Vth:      float = 1.0

    @nn.compact
    def __call__(self, carry, x):
        # Kernels
        in_dims        = x.shape[-1]

        kernel_in  = self.param("kernel_in",  nn.initializers.normal(stddev=1.0), (in_dims,          self.num_neurons), self.dtype)
        kernel_h   = self.param("kernel_h",   nn.initializers.normal(stddev=1.0), (self.num_neurons, self.num_neurons), self.dtype)
        kernel_out = self.param("kernel_out", nn.initializers.normal(stddev=1.0), (self.num_neurons, self.out_dims),    self.dtype)

        # Parameters
        R_in  = self.K_in  * self.Vth * self.tau_Vm                * math.sqrt(2 / in_dims)
        R     = self.K_h   * self.Vth * self.tau_Vm / self.tau_syn * math.sqrt(2 / self.num_neurons)
        R_out = self.K_out                                         * math.sqrt(1 / self.num_neurons)

        alpha_syn = math.exp(-self.dt / self.tau_syn)
        alpha_Vm  = math.exp(-self.dt / self.tau_Vm)
        alpha_out = math.exp(-self.dt / self.tau_out)

        # input layer
        x    = x.astype(self.dtype)
        i_in = R_in * conn_dense(kernel_in, x)

        # SNN layer
        def _snn_step(_carry, _x):
            v_m, i_syn, rate, spike = _carry

            i_spike = R * conn_dense(kernel_h, spike.astype(kernel_h.dtype))
            i_syn   = i_syn * alpha_syn + i_spike
            v_m     = lerp(v_m, i_syn + i_in, alpha_Vm)

            spike = v_m > self.Vth
            v_m   = jnp.where(spike, 0, v_m)

            rate    = lerp(rate, (1 / self.dt) * spike.astype(rate.dtype), alpha_out)

            return (v_m, i_syn, rate, spike), None

        def _snn_get_output(_carry):
            v_m, i_syn, rate, spike = _carry

            return R_out * conn_dense(kernel_out, rate)

        # Stepping
        carry, _ = jax.lax.scan(_snn_step, carry, None, round(self.sim_time / self.dt))
        return carry, _snn_get_output(carry)

    def initial_carry(self, key: jax.random.PRNGKey, batch_size: int):
        v_m   = jnp.zeros((batch_size, self.num_neurons), self.dtype)
        i_syn = jnp.zeros((batch_size, self.num_neurons), self.dtype)
        rate  = jnp.zeros((batch_size, self.num_neurons), self.dtype)
        spike = jnp.zeros((batch_size, self.num_neurons), jnp.bool_)

        if self.rand_init_Vm:
            # Random init Vm to [Vr, Vth]
            v_m = jax.random.uniform(key, (batch_size, self.num_neurons), self.dtype, 0, self.Vth)

        return v_m, i_syn, rate, spike

    def carry_metrics(self, carry):
        v_m, i_syn, rate, spike = carry

        return {
            "spikes_per_ms": jnp.mean(jnp.abs(rate)),
            "avg_i_syn":     jnp.mean(jnp.abs(i_syn))
        }
