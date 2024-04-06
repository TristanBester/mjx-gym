import mujoco
import numpy as np
from mujoco import mjx

from mjxgym.envs.mujoco.constants import XML_PATH
from mjxgym.envs.mujoco.types import State
from mjxgym.interfaces.render import Renderer


class Reacher2DRenderer(Renderer[State]):
    def __init__(self):
        self.mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.renderer = mujoco.Renderer(self.mj_model, height=500, width=500)

        # enable joint visualisation
        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        self.cam = mujoco.MjvCamera()
        # Example on how to set camera configuration
        self.cam.azimuth = 90
        self.cam.elevation = -85
        self.cam.distance = 5
        self.cam.lookat = np.array([0.0, 0.0, 0])

    def render_pixels(self, state: State) -> np.array:
        mj_data = mjx.get_data(self.mj_model, state.mjx_data)
        self.renderer.update_scene(
            data=mj_data, camera=self.cam, scene_option=self.scene_option
        )
        pixels = self.renderer.render()
        return pixels

    def animate_trajectory(self, states: list[State], path: str) -> None:
        pass
