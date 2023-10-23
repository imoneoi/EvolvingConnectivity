import math

import jax
import jax.numpy as jnp
import flax.linen as nn


def lerp(y, x, alpha):
    # Linear interpolation
    # = alpha * y + (1 - alpha) * x
    return x + alpha * (y - x)


def conn_dense(kernel, x):
    # Check dtypes
    assert kernel.dtype == jnp.bool_, "Kernel must be boolean."
    assert x.dtype      != jnp.bool_, "Inputs must not be boolean."

    # matmul
    return jax.lax.dot_general(x, kernel.astype(x.dtype), (((x.ndim - 1,), (0,)), ((), ())))


class ConnSNN(nn.Module):
    """Spiking neural network with connectome only, ExpLIF model"""

    # Network parameters
    out_dims: int
    expected_sparsity: float = 0.5

    num_neurons: int = 256
    excitatory_ratio: float = 0.5

    rand_init_Vm: bool = True

    neuron_dtype: jnp.dtype = jnp.float32
    spike_dtype:  jnp.dtype = jnp.int8

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

    Vr:       float = 0.0
    Vth:      float = 1.0

    @nn.compact
    def __call__(self, carry, x):
        # dummy fixed weights
        self.variable("fixed_weights", "dummy", lambda: None)
        # Kernels
        in_dims        = x.shape[-1]
        num_excitatory = round(self.num_neurons * self.excitatory_ratio)

        kernel_in  = self.param("kernel_in",  nn.initializers.zeros, (2 * in_dims, self.num_neurons),      jnp.bool_)
        kernel_h   = self.param("kernel_h",   nn.initializers.zeros, (self.num_neurons, self.num_neurons), jnp.bool_)
        kernel_out = self.param("kernel_out", nn.initializers.zeros, (self.num_neurons, self.out_dims),    jnp.bool_)

        # Parameters
        R_in  = self.K_in  * self.Vth * self.tau_Vm                * math.sqrt(2 / (self.expected_sparsity * in_dims))
        R     = self.K_h   * self.Vth * self.tau_Vm / self.tau_syn * math.sqrt(2 / (self.expected_sparsity * self.num_neurons))
        R_out = self.K_out                                         * math.sqrt(1 / (self.expected_sparsity * self.num_neurons))

        alpha_syn = math.exp(-self.dt / self.tau_syn)
        alpha_Vm  = math.exp(-self.dt / self.tau_Vm)
        alpha_out = math.exp(-self.dt / self.tau_out)

        # input layer
        x    = x.astype(self.neuron_dtype)
        i_in = R_in * conn_dense(kernel_in, jnp.concatenate([x, -x], axis=-1))

        # SNN layer
        def _snn_step(_carry, _x):
            v_m, i_syn, rate, spike = _carry

            i_spike = R * conn_dense(kernel_h, spike).astype(self.neuron_dtype)
            i_syn   = i_syn * alpha_syn + i_spike
            v_m     = lerp(v_m, self.Vr + i_syn + i_in, alpha_Vm)

            spike_bool           = v_m > self.Vth
            v_m                  = jnp.where(spike_bool, self.Vr, v_m)

            spike_exc, spike_inh = jnp.split(spike_bool.astype(self.spike_dtype), [num_excitatory], axis=-1)
            spike                = jnp.concatenate([spike_exc, -spike_inh], axis=-1)

            rate    = lerp(rate, (1 / self.dt) * spike.astype(rate.dtype), alpha_out)

            return (v_m, i_syn, rate, spike), None

        def _snn_get_output(_carry):
            v_m, i_syn, rate, spike = _carry

            return R_out * conn_dense(kernel_out, rate)

        # Stepping
        carry, _ = jax.lax.scan(_snn_step, carry, None, round(self.sim_time / self.dt))
        return carry, _snn_get_output(carry)

    def initial_carry(self, key: jax.random.PRNGKey, batch_size: int):
        v_m   = jnp.full((batch_size, self.num_neurons), self.Vr, self.neuron_dtype)
        i_syn = jnp.zeros((batch_size, self.num_neurons),         self.neuron_dtype)
        rate  = jnp.zeros((batch_size, self.num_neurons),         self.neuron_dtype)
        spike = jnp.zeros((batch_size, self.num_neurons),         self.spike_dtype)

        if self.rand_init_Vm:
            # Random init Vm to [Vr, Vth]
            v_m = jax.random.uniform(key, (batch_size, self.num_neurons), self.neuron_dtype, self.Vr, self.Vth)

        return v_m, i_syn, rate, spike

    def carry_metrics(self, carry):
        v_m, i_syn, rate, spike = carry

        return {
            "spikes_per_ms": jnp.mean(jnp.abs(rate)),
            "avg_i_syn":     jnp.mean(jnp.abs(i_syn))
        }
