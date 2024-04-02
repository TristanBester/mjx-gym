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

    def last_step_td_target(q_values, time_step, next_timestep):
        td_target = timestep.reward
        return td_target

    def mid_step_td_target(q_values, time_step, next_timestep):
        td_target = timestep.reward + timestep.discount * jnp.max(
            q_values[tuple(next_timestep.observation.agent_pos)]
        )
        return td_target

    td_target = jax.lax.cond(
        timestep.is_last(),
        last_step_td_target,
        mid_step_td_target,
        q_values,
        timestep,
        next_timestep,
    )

    td_error = td_target - curr_val
    updated_val = curr_val + 0.1 * td_error
    q_values = q_values.at[tuple(state_action_idx)].set(updated_val)
    return q_values


@loop_tqdm(N_STEPS)
@jax.jit
def fori_body(_, val):
    (
        q_values,
        key,
        state,
        timestep,
    ) = val

    key, subkey = jax.random.split(key)
    action = jax.random.randint(subkey, (), 0, 4)
    next_state, next_timestep = env.step(state, action)
    q_values = update_q_values(q_values, timestep, action, next_timestep)

    val = (q_values, key, next_state, next_timestep)
    return val


def train_agent():
    env = AutoResetWrapper(GridWorld())
    q_values = jnp.zeros((5, 5, 4))
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    state, timestep = env.reset(subkey)
    val = (q_values, key, state, timestep)
    val = jax.lax.fori_loop(0, N_STEPS, fori_body, val)
    return val[0]


# 722 it/s with jit_step
if __name__ == "__main__":
    env = AutoResetWrapper(GridWorld())
    q_values = train_agent()

    print(q_values.max())

    # env = AutoResetWrapper(GridWorld())
    # key = jax.random.PRNGKey(0)
    # state, timestep = env.reset(key)

    # state, timestep = env.step(state, jnp.array(0))

    # print(timestep)


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
