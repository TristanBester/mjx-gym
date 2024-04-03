import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mediapy import write_video
from numpy.core.multiarray import array as array

from mjxgym.envs.gridworld.types import State
from mjxgym.interfaces.render import Renderer


class GridWorldRenderer(Renderer[State]):
    def __call__(self, state: State) -> None:
        """Render the given environment state."""
        self._draw(state)
        plt.show()

    def render_pixels(self, state: State) -> np.array:
        fig, _ = self._draw(state)
        # Render the plot to a buffer
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        # Convert the buffer to a NumPy array
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (h, w, 4)
        rgb_buf = buf[:, :, 1:]
        return rgb_buf

    def animate_trajectory(self, states: list[State], path: str) -> None:
        frames = []
        for state in states:
            frames.append(self.render_pixels(state))

        write_video(path=path, images=frames, fps=1)

    def _draw(self, state: State) -> tuple[Figure, Axes]:
        fig = Figure()
        ax = fig.gca()
        grid_size = state.grid.shape[0]
        ax.set_xticks(np.arange(-0.5, grid_size - 1, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size - 1, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.invert_yaxis()

        # Render the grid
        for row in range(grid_size):
            for col in range(grid_size):
                tile_value = state.grid[row, col]
                if state.agent_pos[0] == row and state.agent_pos[1] == col:
                    # Render the agent
                    rect = plt.Rectangle([col - 0.5, row - 0.5], 1, 1, color="r")
                    ax.add_patch(rect)
                    ax.text(col, row, "A", ha="center", va="center")
                elif row == grid_size - 1 and col == grid_size - 1:
                    # Render the goal
                    rect = plt.Rectangle([col - 0.5, row - 0.5], 1, 1, color="g")
                    ax.add_patch(rect)
                    ax.text(col, row, "G", ha="center", va="center")
                elif tile_value == 0:
                    # Render the empty tile
                    rect = plt.Rectangle([col - 0.5, row - 0.5], 1, 1, color="white")
                    ax.add_patch(rect)
                else:
                    # Render the wall
                    rect = plt.Rectangle([col - 0.5, row - 0.5], 1, 1, color="black")
                    ax.add_patch(rect)
        return fig, ax
