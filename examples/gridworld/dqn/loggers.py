from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


class Logger:
    def __init__(self):
        self.metric_buffers = defaultdict(list)

    def log(self, metrics):

        print(metrics["reward"])
        print(metrics["done"])

        # for i, x in metrics.items():
        #     self.metric_buffers[i].append(x)

    def dump(self):
        print("Dumping...")

        # for i, x in self.metric_buffers.items():
        #     print(f"{i}: {x}")


# class LogAggregator:
#     def __init__(self):
#         self.loggers = []

#     def add_logger(self, logger):
#         self.loggers.append(logger)

#     def log(self, metrics):
#         for logger in self.loggers:
#             logger.log(metrics)

#     def dump(self):
#         for logger in self.loggers:
#             logger.dump()


class ReturnsLogger:
    def __init__(self, env_count):
        self.env_count = env_count
        self.curr_ep_returns = np.zeros(env_count)
        self.returns_buffer = {i: [] for i in range(env_count)}

    def log(self, rewards, dones):
        self.curr_ep_returns += rewards

        # print(rewards)
        # print(dones)
        # print(self.curr_ep_returns)
        # print()

        if np.any(dones):
            # print("Episode done")
            done_idx = np.argwhere(dones).flatten()
            # print(done_idx)
            for i in done_idx:
                self.returns_buffer[int(i)].append(float(self.curr_ep_returns[i]))
                self.curr_ep_returns[i] = 0.0

        # print(self.returns_buffer)
        # print(self.curr_ep_returns)

    def dump(self):
        print("Dumping...")

        for i, x in self.returns_buffer.items():
            plt.plot(x)
        plt.show()
