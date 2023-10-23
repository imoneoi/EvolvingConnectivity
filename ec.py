from functools import partial
from typing import Any, Dict, Tuple
import time
import builtins

import jax
import jax.numpy as jnp

import flax
import optax

from brax import envs
from brax.training.acme import running_statistics
from brax.training.acme import specs

from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
import optuna

from networks import NETWORKS
from utils.functions import mean_weight_abs, finitemean, save_obj_to_file


# Use RBG generator for less memory consumption
# Default RNG needs 2*N extra memory, while RBG needs none, when generating array with size N
# https://jax.readthedocs.io/en/latest/jax.random.html
jax.config.update("jax_default_prng_impl", "unsafe_rbg")

# Hack for resolving bfloat16 pickling issue https://github.com/google/jax/issues/8505
builtins.bfloat16 = jnp.dtype("bfloat16").type


@flax.struct.dataclass
class ESConfig:
    # Network, optim & env class
    network_cls: Any = None
    optim_cls:   Any = None
    env_cls:     Any = None

    # [Hyperparameters] ES
    pop_size:       int = 10240
    lr:           float = 0.15

    eps:          float = 1e-3

    weight_decay: float = 0.    # For sparsity regularization

    # [Hyperparameters] Warmup
    warmup_steps:   int = 0

    # [Hyperparameters] Eval
    eval_size:      int = 128

    # [Computing] Data types
    action_dtype: Any   = jnp.float32  # brax uses fp32

    p_dtype:       Any  = jnp.float32
    network_dtype: Any  = jnp.float32


@flax.struct.dataclass
class RunnerState:
    key: Any
    # Normalizer
    normalizer_state: running_statistics.RunningStatisticsState
    # Env reset state pool
    env_reset_pool: Any
    # Network optimization
    params:        Any
    fixed_weights: Any
    opt_state:     Any


@flax.struct.dataclass
class PopulationState:
    # Network
    network_params: Any
    network_states: Any
    # Env
    env_states:     Any
    # Fitness
    fitness_totrew: jnp.ndarray
    fitness_sum:    jnp.ndarray
    fitness_n:      jnp.ndarray


def _centered_rank_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Centered rank from: https://arxiv.org/pdf/1703.03864.pdf"""

    shape = x.shape
    x     = x.ravel()

    x = jnp.argsort(jnp.argsort(x))
    x = x / (len(x) - 1) - .5
    return x.reshape(shape)


def _sample_bernoulli_parameter(key: Any, params: Any, sampling_dtype: Any, batch_size: Tuple = ()) -> Any:
    """Sample parameters from Bernoulli distribution. """

    num_vars = len(jax.tree_util.tree_leaves(params))
    treedef = jax.tree_util.tree_structure(params)

    all_keys = jax.random.split(key, num=num_vars)
    noise = jax.tree_util.tree_map(
        lambda p, k: jax.random.uniform(k, (*batch_size, *p.shape), sampling_dtype) < p,
        params, jax.tree_util.tree_unflatten(treedef, all_keys))

    return noise


def _deterministic_bernoulli_parameter(params: Any, batch_size: Tuple = ()) -> Any:
    """Deterministic evaluation, using p > 0.5 as True, p <= 0.5 as False"""

    return jax.tree_util.tree_map(lambda p: jnp.broadcast_to(p > 0.5, (*batch_size, *p.shape)), params)


# Evaluate the population for a single step
def _evaluate_step(pop: PopulationState, runner: RunnerState, conf: ESConfig) -> PopulationState:
    # step env
    # NOTE: vmapping apply for multiple set of parameters (broadcast fixed weights)
    vmapped_apply = jax.vmap(conf.network_cls.apply, ({"params": 0, "fixed_weights": None}, 0, 0))

    obs_norm                = running_statistics.normalize(pop.env_states.obs, runner.normalizer_state)
    new_network_states, act = vmapped_apply({"params": pop.network_params, "fixed_weights": runner.fixed_weights}, pop.network_states, obs_norm)
    assert act.dtype == conf.network_dtype   # Sanity check, avoid silent promotion

    act = jnp.clip(act, -1, 1)  # brax do not clip actions internally.

    # NOTE: Cast type and avoid NaNs, set them to 0
    if act.dtype != conf.action_dtype:
        act = jnp.where(jnp.isnan(act), 0, act).astype(conf.action_dtype)

    new_env_states = conf.env_cls.step(pop.env_states, act)

    # calculate episodic rewards
    new_fitness_totrew = pop.fitness_totrew + new_env_states.reward

    new_fitness_sum    = jnp.where(new_env_states.done, pop.fitness_sum + new_fitness_totrew, pop.fitness_sum)
    new_fitness_n      = jnp.where(new_env_states.done, pop.fitness_n   + 1,                  pop.fitness_n)
    # clear tot rew
    new_fitness_totrew = jnp.where(new_env_states.done, 0, new_fitness_totrew)

    # reset done envs
    # Reference: brax / envs / wrapper.py
    def _where_done(x, y):
        done = new_env_states.done
        done = done.reshape([-1] + [1] * (len(x.shape) - 1))
        return jnp.where(done, x, y)

    new_env_states = jax.tree_map(_where_done, runner.env_reset_pool, new_env_states)

    return pop.replace(
        # Network
        network_states=new_network_states,
        # Env
        env_states=new_env_states,
        # Fitness
        fitness_totrew=new_fitness_totrew,
        fitness_sum=new_fitness_sum,
        fitness_n=new_fitness_n
    )


@partial(jax.jit, static_argnums=(2,))
def _runner_init(key: Any, network_init_key: Any, conf: ESConfig) -> RunnerState:
    # split run keys for initializing env
    key, env_init_key = jax.random.split(key)

    # init env
    env_reset_pool = conf.env_cls.reset(jax.random.split(env_init_key, conf.pop_size))

    # init network params + opt state
    network_variables = jax.jit(conf.network_cls.init, donate_argnums=(1,))(
        {"params": network_init_key, "fixed_weights": network_init_key},
        conf.network_cls.initial_carry(jax.random.PRNGKey(0), conf.pop_size),
        env_reset_pool.obs
    )
    network_params = network_variables["params"]
    network_fixed_weights = network_variables["fixed_weights"]

    # set params to p=0.5 Bernoulli distribution
    network_params = jax.tree_map(lambda x: jnp.full_like(x, 0.5, conf.p_dtype), network_params)
    optim_state = conf.optim_cls.init(network_params)

    # runner state
    runner = RunnerState(
        key=key,
        normalizer_state=running_statistics.init_state(specs.Array((conf.env_cls.observation_size, ), jnp.float32)),
        env_reset_pool=env_reset_pool,

        params=network_params,
        fixed_weights=network_fixed_weights,
        opt_state=optim_state
    )
    return runner


@partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
def _runnner_run(runner: RunnerState, conf: ESConfig) -> Tuple[RunnerState, Dict]:
    metrics = {}

    # split keys for this run
    new_key, run_key, carry_key = jax.random.split(runner.key, 3)
    runner = runner.replace(key=new_key)

    # Generate params with bernoulli distribution
    train_params = _sample_bernoulli_parameter(run_key, runner.params, conf.network_dtype, (conf.pop_size - conf.eval_size, ))
    eval_params  = _deterministic_bernoulli_parameter(runner.params, (conf.eval_size, ))

    network_params = jax.tree_map(lambda train, eval: jnp.concatenate([train, eval], axis=0), train_params, eval_params)

    # Split the eval and train fitness, returns [fitness, eval_fitness]
    def _split_fitness(x):
        return jnp.split(x, [conf.pop_size - conf.eval_size, ])

    # Initialize population
    pop = PopulationState(
        # Network
        network_params=network_params,
        network_states=conf.network_cls.initial_carry(carry_key, conf.pop_size),
        # Env
        env_states=runner.env_reset_pool,
        # Fitness
        fitness_totrew=jnp.zeros(conf.pop_size),
        fitness_sum=jnp.zeros(conf.pop_size),
        fitness_n=jnp.zeros(conf.pop_size, dtype=jnp.int32)
    )

    # (PNN) Run some steps to warm up states
    if conf.warmup_steps > 0:
        pop, _ = jax.lax.scan(lambda p, x: (_evaluate_step(p, runner, conf), None), pop, None, length=conf.warmup_steps)

        warmup_fitness, warmup_eval_fitness = _split_fitness(pop.fitness_sum / pop.fitness_n)
        metrics.update({
            "warmup_fitness":      finitemean(warmup_fitness),
            "warmup_eval_fitness": finitemean(warmup_eval_fitness)
        })

        # (PNN) Update normalizer using warmup data
        runner = runner.replace(normalizer_state=running_statistics.update(runner.normalizer_state, pop.env_states.obs))
        # (PNN) Reset envs + Clear fitness
        pop = pop.replace(
            # Env
            env_states=runner.env_reset_pool,
            # Fitness
            fitness_totrew=jnp.zeros(conf.pop_size),
            fitness_sum=jnp.zeros(conf.pop_size),
            fitness_n=jnp.zeros(conf.pop_size, dtype=jnp.int32)
        )

    # Evaluate
    def _eval_stop_cond(p: PopulationState) -> jnp.ndarray:
        # Stop when finished
        return ~jnp.all(p.fitness_n >= 1)

    pop = jax.lax.while_loop(_eval_stop_cond, (lambda p: _evaluate_step(p, runner, conf)), pop)

    # Update normalizer using terminal states
    # FIXME: May be biased towards states near episode terminal
    if conf.warmup_steps <= 0:
        runner = runner.replace(normalizer_state=running_statistics.update(runner.normalizer_state, pop.env_states.obs))

    # Calculate population metrics
    if hasattr(conf.network_cls, "carry_metrics"):
        metrics.update(conf.network_cls.carry_metrics(pop.network_states))

    # Calculate fitness
    fitness, eval_fitness = _split_fitness(pop.fitness_sum / pop.fitness_n)

    # Reconstruct noise using network parameters
    # NOTE: use -grads to do gradient ascent
    weight = _centered_rank_transform(fitness)
    def _nes_grad(p, theta):
        w = weight.reshape((-1,) + (1,) * (theta.ndim - 1)).astype(p.dtype)

        return -jnp.mean(w * (theta - p), axis=0)

    grads = jax.tree_map(lambda p, theta: _nes_grad(p, theta[:(conf.pop_size - conf.eval_size)]), runner.params, pop.network_params)

    # Gradient step
    updates, new_opt_state = conf.optim_cls.update(grads, runner.opt_state, runner.params)
    new_params = optax.apply_updates(runner.params, updates)

    # Clip to Bernoulli range with exploration
    new_params = jax.tree_map(lambda p: jnp.clip(p, conf.eps, 1 - conf.eps), new_params)

    runner = runner.replace(
        params=new_params,
        opt_state=new_opt_state
    )

    # Metrics
    metrics.update({
        "fitness":        jnp.mean(fitness),
        "eval_fitness":   jnp.mean(eval_fitness),

        "sparsity": mean_weight_abs(new_params)
    })
    return runner, metrics


def main(conf):
    conf = OmegaConf.merge({
        # Task
        "seed": 0,
        "task": "humanoid",
        "task_conf": {
        },
        "episode_conf": {
            "max_episode_length": 1000,
            "action_repeat": 1
        },

        # Train & Checkpointing
        "total_generations": 1000,
        "save_every": 50,

        # Network
        "network_type": "ConnSNN",
        "network_conf": {},

        # ES hyperparameter (see ESConfig)
        "es_conf": {}
    }, conf)
    # Naming
    conf = OmegaConf.merge({
        "project_name": f"E-SNN-{conf.task}",
        "run_name":     f"EC {conf.seed} {conf.network_type} {time.strftime('%H:%M %m-%d')}"
    }, conf)
    # ES Config
    es_conf = ESConfig(**conf.es_conf)

    print(OmegaConf.to_yaml(conf))
    print(es_conf)

    # create env cls
    env = envs.get_environment(conf.task, **conf.task_conf)
    env = envs.wrappers.EpisodeWrapper(env, conf.episode_conf.max_episode_length, conf.episode_conf.action_repeat)
    env = envs.wrappers.VmapWrapper(env)

    # create network cls
    network_cls = NETWORKS[conf.network_type]
    network     = network_cls(
        out_dims=env.action_size,
        neuron_dtype=es_conf.network_dtype,
        **conf.network_conf
    )

    # create optim cls
    optim = optax.chain(
        optax.scale_by_adam(mu_dtype=es_conf.p_dtype),
        (optax.add_decayed_weights(es_conf.weight_decay) if es_conf.weight_decay > 0 else optax.identity()),
        optax.scale(-es_conf.lr)
    )

    # [initialize]
    # initialize cls in es conf
    es_conf = es_conf.replace(
        network_cls=network,
        optim_cls=optim,
        env_cls=env
    )

    # runner state
    key_run, key_network_init = jax.random.split(jax.random.PRNGKey(conf.seed))
    runner = _runner_init(key_run, key_network_init, es_conf)

    # save model path
    conf.save_model_path = "models/{}/{}/".format(conf.project_name, conf.run_name)

    # wandb
    if "log_group" in conf:
        wandb.init(reinit=True, project=f"(G) E-SNN-{conf.task}", group=conf.log_group, name=str(conf.seed), config=OmegaConf.to_container(conf))
    else:
        wandb.init(reinit=True, project=conf.project_name, name=conf.run_name, config=OmegaConf.to_container(conf))

    # run
    for step in tqdm(range(1, conf.total_generations + 1)):
        runner, metrics = _runnner_run(runner, es_conf)

        metrics = jax.device_get(metrics)
        wandb.log(metrics, step=step)

        if not (step % conf.save_every):
            fn = conf.save_model_path + str(step)
            save_obj_to_file(fn, dict(
                conf=conf,
                state=dict(
                    normalizer_state=runner.normalizer_state,
                    fixed_weights=runner.fixed_weights,
                    params=runner.params
                )
            ))

            wandb.save(fn)

    return metrics


def sweep(seed: int, conf_override: OmegaConf):
    def _objective(trial: optuna.Trial):
        conf = OmegaConf.merge(conf_override, {
            "seed": seed * 1000 + trial.number,

            "project_name": f"E-SNN-sweep",

            "es_conf": {
                "lr":           trial.suggest_categorical("lr",  [0.01, 0.05, 0.1, 0.15, 0.2]),
                "eps":          trial.suggest_categorical("eps", [1e-4, 1e-3, 0.01, 0.1, 0.2]),
            },
            "network_conf": {
                "num_neurons":  trial.suggest_categorical("num_neurons", [128, 256]),
            }
        })

        metrics = main(conf)
        return metrics["eval_fitness"]

    optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=seed)).optimize(_objective)


if __name__ == "__main__":
    _config = OmegaConf.from_cli()
    if hasattr(_config, "sweep"):
        sweep(_config.sweep, _config)
    else:
        main(_config)
