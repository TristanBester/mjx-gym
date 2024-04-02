from functools import partial

import jax
import jax.numpy as jnp
from jax_tqdm import loop_tqdm

from mjxgym.env import GridWorld
from mjxgym.wrappers import AutoResetWrapper

N_STEPS = 1_000_000


@jax.jit
def update_q_values(q_values, timestep, action, next_timestep):
    state_action_idx = jnp.append(timestep.observation.agent_pos, action)
    curr_val = q_values[tuple(state_action_idx)]

    # This handles the last step as discount is 0 if next_step is terminal
    td_target = next_timestep.reward + next_timestep.discount * jnp.max(
        q_values[tuple(next_timestep.observation.agent_pos)]
    )

    td_error = td_target - curr_val
    updated_val = curr_val + 0.1 * td_error
    q_values = q_values.at[tuple(state_action_idx)].set(updated_val)
    return q_values


# @loop_tqdm(N_STEPS)
@jax.jit
def fori_body(_, val):
    (
        q_values,
        key,
        state,
        timestep,
    ) = val

    key, eps_key, act_key = jax.random.split(key, 3)
    eps = jax.random.uniform(eps_key, ())
    all_zero = jnp.max(q_values[tuple(state.agent_pos)]) == 0.0
    eps_below = eps < 0.1
    select_random = all_zero | eps_below
    action = jax.lax.cond(
        select_random,
        lambda: jax.random.randint(act_key, (), 0, 4),
        lambda: jnp.argmax(q_values[tuple(state.agent_pos)]),
    )

    next_state, next_timestep = env.step(state, action)
    q_values = update_q_values(q_values, timestep, action, next_timestep)
    val = (q_values, key, next_state, next_timestep)
    return val


def train_agent(key):
    env = AutoResetWrapper(GridWorld())
    q_values = jnp.zeros((5, 5, 4))
    key, subkey = jax.random.split(key)

    state, timestep = env.reset(subkey)
    val = (q_values, key, state, timestep)
    val = jax.lax.fori_loop(0, N_STEPS, fori_body, val)
    return val[0]


if __name__ == "__main__":
    env = AutoResetWrapper(GridWorld())
    key = jax.random.PRNGKey(0)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state, timestep = jit_reset(key)
    state, timestep = jit_step(state, jnp.array(0))

    print(state)
    print(timestep)

    # q_values = train_agent(key)
    # print(q_values.max())

    # for row in range(5):
    #     for col in range(5):
    #         print(f"{jnp.max(q_values[row, col, :]):.3f}", end=" ")
    #     print()

    # print()
    # directions = ["→", "←", "↑", "↓"]
    # for row in range(5):
    #     for col in range(5):
    #         mx_val = jnp.max(q_values[row, col, :])
    #         d = jnp.argmax(q_values[row, col, :])

    #         if mx_val == 0:
    #             print("x", end="  ")
    #         else:
    #             print(directions[d], end="  ")
    #     print()

    #####

    # vmap_train_agents = lambda key, n_agents: jax.vmap(train_agent)(
    #     jax.random.split(key, n_agents)
    # )

    # q_values = vmap_train_agents(key, 4)
    # print(q_values.shape)
    # print(jnp.max(q_values, axis=(1, 2, 3)))

    # vmap_train_agents = lambda key, n_agents: jax.vmap(train_agent)(
    #     jax.random.split(key, n_agents)
    # )

    # pmap_train_agents = lambda key, n_agents: jax.pmap(vmap_train_agents)(
    #     jax.random.split(key, 2), jnp.array([2, 2])
    # )

    # n_runs = 10
    # n_cores = 6
    # q_values = pmap_train_agents(key, 10)
    # print(q_values.shape)
    # print(jnp.max(q_values, axis=(1, 2, 3)))

    # f = jax.pmap(jax.vmap(train_agent))

    # keys = jax.random.split(key, 4)
    # f(keys)

    # keys = jnp.asarray(jax.random.split(key, 1000))

    # # print(keys)
    # # print(keys.reshape(2, -1, 2).shape)
    # # print(keys.reshape(2, -1, 2))

    # # 2 devices, 5 keys per device, key shape is (2,)
    # key_batches = keys.reshape(8, -1, 2)

    # key_batch = key_batches[0]

    # # print(key_batch)

    # vmap_train = jax.vmap(train_agent)

    # # q_values = vmap_train(key_batch)
    # # print(q_values.shape)
    # # print(jnp.max(q_values, axis=(1, 2, 3)))

    # pmap_train = jax.pmap(vmap_train)

    # import time

    # t_0 = time.time()
    # q_values = pmap_train(key_batches)
    # t_1 = time.time()
    # print("Elapsed time:", t_1 - t_0)

    # print(q_values.shape)

    # q_values = jnp.concatenate(q_values, axis=0)

    # print(q_values.shape)
    # print(jnp.max(q_values, axis=(1, 2, 3)))

    # q_values = q_values[0]
    # for row in range(5):
    #     for col in range(5):
    #         print(f"{jnp.max(q_values[row, col, :]):.3f}", end=" ")
    #     print()

    # print()
    # directions = ["→", "←", "↑", "↓"]
    # for row in range(5):
    #     for col in range(5):
    #         mx_val = jnp.max(q_values[row, col, :])
    #         d = jnp.argmax(q_values[row, col, :])

    #         if mx_val == 0:
    #             print("x", end="  ")
    #         else:
    #             print(directions[d], end="  ")
    #     print()

    # keys = jax.random.split(key, 10)
    # q_values = jax.soft_pmap(train_agent)(keys)
    # print(q_values.shape)
    # print(jnp.max(q_values, axis=(1, 2, 3)))

    # vmap_train = jax.vmap(train_agent)
    # q_values = vmap_train(keys)

    # print(q_values.shape)
    # print(jnp.max(q_values, axis=(1, 2, 3)))

    # pmap_train = jax.pmap(train_agent)
    # q_values = pmap_train(keys)

    # print(q_values.shape)
    # print(jnp.max(q_values, axis=(1, 2, 3)))

    # env = AutoResetWrapper(GridWorld(), next_obs_in_extras=False)
    # key = jax.random.PRNGKey(0)
    # state, timestep = env.reset(key)
    # q_values = jnp.zeros((5, 5, 4))

    # state = state._replace(agent_pos=jnp.array([4, 3]))
    # action = jnp.array(0)

    # val = (q_values, key, state, timestep)
    # fori_body(0, val)

    # env = AutoResetWrapper(GridWorld(), next_obs_in_extras=False)
    # key = jax.random.PRNGKey(0)
    # state, timestep = env.reset(key)

    # # print(timestep)
    # # print()

    # state, timestep = env.step(state, jnp.array(0))

    # print(timestep)
    # print()

    # state, timestep = env.step(state, jnp.array(0))

    # print(timestep)
    # print()


# jit_reset = jax.jit(env.reset)
# jit_step = jax.jit(env.step)
# jit_step(state, jnp.array(1))

# for i in trange(100000):
#     action =
#     next_state, next_timestep = jit_step(state, action)
#     q_values = update_q_values(q_values, timestep, action, next_timestep)
#     state = next_state
#     timestep = next_timestep

#     if i % 1000 == 0:
#         print(q_values.max())
# print(q_values)
