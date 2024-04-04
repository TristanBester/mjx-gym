from typing import Callable

import chex
import flashbax as fbx
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax_tqdm import loop_tqdm

from mjxgym.envs.gridworld.env import GridWorld
from mjxgym.envs.gridworld.generator import GridWorldGenerator
from mjxgym.wrappers.wrappers import VmapAutoResetWrapper


class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class DqnTrainState(TrainState):
    target_network_params: flax.core.FrozenDict
    timesteps: int
    n_updates: int


@jax.jit
def create_train():
    N_ITERS = 1_000_000
    generator = GridWorldGenerator(approx_grid_size=5)
    env = VmapAutoResetWrapper(GridWorld(generator, max_steps=100))

    def train(key: chex.Array):
        """Pass in a batch of keys one for each vectorised environment."""
        # Create and initialise replay buffer.
        buffer = fbx.make_flat_buffer(
            max_length=10000,
            min_length=32,
            sample_batch_size=32,
            add_batch_size=4,
        )
        _, timestep = env.unwrapped.reset(jax.random.PRNGKey(0))
        buffer_state = buffer.init(timestep)

        # Create and initialise Q-network.
        q_network = QNetwork(action_dim=4)
        init_x = timestep.observation.agent_pos
        network_params = q_network.init(key, init_x)
        target_network_params = jax.tree_map(lambda x: jnp.copy(x), network_params)

        # Create optimizer
        tx = optax.adam(learning_rate=0.0001)

        train_state = DqnTrainState.create(
            apply_fn=q_network.apply,
            params=network_params,
            target_network_params=target_network_params,
            tx=tx,
            timesteps=0,
            n_updates=0,
        )

        @jax.vmap
        def eps_greedy(key, q_values):
            eps_key, act_key = jax.random.split(key)
            eps = 0.1
            explore = jax.random.uniform(eps_key) < eps
            actions = jax.lax.cond(
                explore,
                lambda key: jax.random.randint(key, shape=(), minval=0, maxval=4),
                lambda _: jnp.argmax(q_values),
                act_key,
            )
            return actions

        @loop_tqdm(N_ITERS)
        def fori_body(i: int, val: tuple) -> tuple:
            key, train_state, buffer_state, states, timesteps = val

            # Sample actions from the Q-network
            key, subkey = jax.random.split(key)
            act_keys = jax.random.split(subkey, 4)
            q_values = q_network.apply(
                train_state.params, timesteps.observation.agent_pos
            )
            actions = eps_greedy(act_keys, q_values)

            # Execute actions in the environments
            next_states, next_timesteps = env.step(states, actions)
            buffer_state = buffer.add(buffer_state, next_timesteps)
            train_state = train_state.replace(
                timesteps=train_state.timesteps + 4,
            )

            def learn(key, train_state):
                key, subkey = jax.random.split(key)
                batch = buffer.sample(buffer_state, subkey).experience
                q_values_target = q_network.apply(
                    train_state.target_network_params,
                    batch.second.observation.agent_pos,
                )
                max_q_values_target = jnp.max(q_values_target, axis=-1)
                td_target = (
                    batch.second.reward + batch.second.discount * max_q_values_target
                )

                def loss_fn(params):
                    q_values = q_network.apply(
                        params, batch.first.observation.agent_pos
                    )
                    selected_actions = batch.second.action
                    selected_actions = jnp.expand_dims(selected_actions, axis=-1)
                    selected_action_q_values = jnp.take_along_axis(
                        q_values, selected_actions, axis=-1
                    ).flatten()
                    td_error = jnp.mean((selected_action_q_values - td_target) ** 2)
                    return td_error

                loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    n_updates=train_state.n_updates + 1,
                )
                return train_state, loss

            buffer_ready = buffer.can_sample(buffer_state)
            learning_started = train_state.timesteps > 1000
            update_interval = i % 1 == 0
            train_network = buffer_ready & learning_started & update_interval

            train_network = True

            key, subkey = jax.random.split(key)

            # FIXME: Returning zero loss for identity is not valid
            train_state, loss = jax.lax.cond(
                train_network,
                lambda key, ts: learn(key, ts),
                lambda _, ts: (ts, jnp.array(0.0)),
                subkey,
                train_state,
            )

            update_target = train_state.n_updates % 1 == 0

            def update_target_network(train_state):
                updated_target_params = optax.incremental_update(
                    train_state.params, train_state.target_network_params, 0.1
                )
                train_state = train_state.replace(
                    target_network_params=updated_target_params
                )
                return train_state

            train_state = jax.lax.cond(
                update_target, update_target_network, lambda ts: ts, train_state
            )

            return (
                key,
                train_state,
                buffer_state,
                next_states,
                next_timesteps,
            )

        # Initialise the environment
        key, subkey = jax.random.split(key)
        reset_keys = jax.random.split(subkey, 4)
        states, timesteps = env.reset(reset_keys)
        buffer_state = buffer.add(buffer_state, timesteps)
        train_state = train_state.replace(timesteps=train_state.timesteps + N_ENVS)
        val = (
            key,
            train_state,
            buffer_state,
            states,
            timesteps,
        )
        (
            _,
            train_state,
            _,
            _,
            _,
        ) = jax.lax.fori_loop(0, N_ITERS, fori_body, val)
        return train_state

    N_ENVS = 4
    key = jax.random.PRNGKey(10)
    return train(key)


if __name__ == "__main__":
    train_state = create_train()
    q_network = QNetwork(action_dim=4)

    directions = ["→", "←", "↑", "↓"]
    for row in range(5):
        for col in range(5):
            q_values = q_network.apply(train_state.params, jnp.array([row, col]))
            action = directions[jnp.argmax(q_values)]

            print(action, end=" ")
        print()
