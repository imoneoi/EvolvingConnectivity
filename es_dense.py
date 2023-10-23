from functools import partial
from typing import Any, Dict, Tuple
import time

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

from networks import NETWORKS
from utils.functions import rand_normal_like_tree, zeros_like_tree, finitemean, param_norm, save_obj_to_file


# Use RBG generator for less memory consumption
# Default RNG needs 2*N extra memory, while RBG needs none, when generating array with size N
# https://jax.readthedocs.io/en/latest/jax.random.html
jax.config.update("jax_default_prng_impl", "rbg")


@flax.struct.dataclass
class ESConfig:
    # Network, optim & env class
    network_cls: Any = None
    optim_cls:   Any = None
    env_cls:     Any = None
    # [Hyperparameters] ES
    pop_size:       int = 10240
    sigma:        float = 0.02
    lr:           float = 0.01
    weight_decay: float = 0.1

    # [Hyperparameters] Warmup
    warmup_steps:   int = 0

    # [Hyperparameters] Eval
    eval_size:      int = 128


@flax.struct.dataclass
class RunnerState:
    key: Any
    # Normalizer
    normalizer_state: running_statistics.RunningStatisticsState
    # Env reset state pool
    env_reset_pool: Any
    # Network optimization
    params:    Any
    opt_state: Any


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


# Evaluate the population for a single step
def _evaluate_step(pop: PopulationState, runner: RunnerState, conf: ESConfig) -> PopulationState:
    # step env
    # NOTE: vmapping apply for multiple set of parameters
    obs_norm                = running_statistics.normalize(pop.env_states.obs, runner.normalizer_state)
    new_network_states, act = jax.vmap(conf.network_cls.apply)(pop.network_params, pop.network_states, obs_norm)

    act = jnp.clip(act, -1, 1)  # brax do not clip actions internally.
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
    network_params = jax.jit(conf.network_cls.init, donate_argnums=(1,))(
        network_init_key,
        conf.network_cls.initial_carry(jax.random.PRNGKey(0), conf.pop_size),
        env_reset_pool.obs
    )
    optim_state = conf.optim_cls.init(network_params)

    # runner state
    runner = RunnerState(
        key=key,
        normalizer_state=running_statistics.init_state(specs.Array((conf.env_cls.observation_size, ), jnp.float32)),
        env_reset_pool=env_reset_pool,
        params=network_params,
        opt_state=optim_state
    )
    return runner


@partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
def _runnner_run(runner: RunnerState, conf: ESConfig) -> Tuple[RunnerState, Dict]:
    metrics = {}

    # split keys for this run
    new_key, run_key, carry_key = jax.random.split(runner.key, 3)
    runner = runner.replace(key=new_key)

    # Generate params with antithetic noise
    # params: [original (for eval), pos noise, neg noise]
    noise          = rand_normal_like_tree(run_key, runner.params, std=conf.sigma, batch_shape=((conf.pop_size - conf.eval_size) // 2, ))
    zeros          = zeros_like_tree(runner.params, batch_shape=(conf.eval_size, ))
    network_params = jax.tree_map(lambda x, n, z: x + jnp.concatenate([n, -n, z], axis=0), runner.params, noise, zeros)

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

    # Transform and split antithetic fitness --> weight
    # weight: pop_size / 2
    weight = _centered_rank_transform(fitness)
    weight_pos, weight_neg = jnp.split(weight, 2, axis=-1)
    weight = weight_pos - weight_neg

    # Reconstruct noise using first half of network parameters
    # NOTE: grads should be divided by 2 * sigma to be mathematically correct, but as we use AdamW, it has no effects
    # NOTE: use -grads to do gradient ascent
    grads = jax.tree_map(lambda p, op: -jnp.mean(weight.reshape([-1] + [1] * (p.ndim - 1)) * (p[:(conf.pop_size - conf.eval_size) // 2] - op), axis=0),
                         pop.network_params, runner.params)

    # Gradient step
    updates, new_opt_state = conf.optim_cls.update(grads, runner.opt_state, runner.params)
    new_params = optax.apply_updates(runner.params, updates)

    runner = runner.replace(
        params=new_params,
        opt_state=new_opt_state
    )

    # Metrics
    metrics.update({
        "fitness":        jnp.mean(fitness),
        "eval_fitness":   jnp.mean(eval_fitness),

        "param_norm": param_norm(new_params)
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
        "network_type": "DenseLSTM",
        "network_conf": {},

        # ES hyperparameter (see ESConfig)
        "es_conf": {}
    }, conf)
    # Naming
    conf = OmegaConf.merge({
        "project_name": f"ESDense-{conf.task}",
        "run_name":     f"ES {conf.seed} {conf.network_type} {time.strftime('%H:%M %m-%d')}"
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
        **conf.network_conf
    )

    # create optim cls
    optim       = optax.adamw(learning_rate=es_conf.lr, weight_decay=es_conf.weight_decay)

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
        wandb.init(reinit=True, project=f"(G) ESDense-{conf.task}", group=conf.log_group, name=str(conf.seed), config=OmegaConf.to_container(conf))
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
                    params=runner.params
                )
            ))

            wandb.save(fn)


if __name__ == "__main__":
    main(OmegaConf.from_cli())
