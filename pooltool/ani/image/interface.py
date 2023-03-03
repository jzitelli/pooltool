from typing import Any, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray
from panda3d.core import ClockObject, FrameBufferProperties, GraphicsOutput, Texture

from pooltool.ani.animate import Interface, ShowBaseConfig
from pooltool.ani.camera import CameraState, cam, camera_states
from pooltool.ani.globals import Global
from pooltool.ani.hud import HUDElement, hud
from pooltool.system.datatypes import System, multisystem
from pooltool.system.render import visual

DEFAULT_FBP = FrameBufferProperties()
DEFAULT_FBP.setRgbColor(True)
DEFAULT_FBP.setRgbaBits(8, 8, 8, 0)
DEFAULT_FBP.setDepthBits(24)

DEFAULT_SHOWBASE_CONFIG = ShowBaseConfig(
    window_type="offscreen",
    monitor=False,
    fb_prop=DEFAULT_FBP,
)

DEFAULT_CAMERA = camera_states["7_foot_offcenter"]


def _resize_window(size):
    """Changes window size when provided the dimensions (x, y) in pixels"""
    Global.base.win.setSize(*[int(dim) for dim in size])


class Exporter(Protocol):
    def save(self, img: NDArray[np.uint8]) -> Any:
        ...


class ImageSaver(Interface):
    """An interface for saving shots as series of images"""

    def __init__(self, config: ShowBaseConfig = DEFAULT_SHOWBASE_CONFIG):
        Interface.__init__(self, config=config)

        # Create and setup the image texture
        self.tex = Texture()
        Global.base.win.addRenderTexture(
            self.tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPColor
        )

        # Aim to render 1000 FPS so the clock doesn't sleep between frames
        Global.clock.setMode(ClockObject.MLimited)
        Global.clock.setFrameRate(1000)

    def get_image_array(self) -> NDArray[np.uint8]:
        """Return array of current image texture"""
        assert self.tex.hasRamImage()

        array = np.frombuffer(self.tex.getRamImage(), dtype=np.uint8)
        array.shape = (
            self.tex.getYSize(),
            self.tex.getXSize(),
            self.tex.getNumComponents(),
        )

        # This flips things rightside up and orders RGB correctly
        return array[::-1, :, ::-1]

    def save(
        self,
        shot: System,
        exporter: Exporter,
        camera_state: CameraState = DEFAULT_CAMERA,
        size: Tuple[int, int] = (230, 144),
        show_hud: bool = False,
        fps: float = 30.0,
    ):
        """Save a shot as a series of images

        Args:
            shot:
                The shot you would like visualized. It should already by simulated. It
                is OK if you have continuized the shot (you can check with
                shot.continuized), but the continuization will be overwritten to match
                the `fps` chosen in this method.
            exporter:
                The exporter used to save the shot frames.
            camera_state:
                A camera state specifying the camera's view of the table.
            size:
                The number of pixels in x and y. If x:y != 1.6, the aspect ratio will
                look distorted.
            show_hud:
                If True, the HUD will appear in the images.
            fps:
                This is the rate (in frames per second) that an image of the shot is
                taken.
        """
        shot.continuize(dt=1 / fps)

        multisystem.reset()
        multisystem.append(shot)

        _resize_window(size)

        self.create_scene()

        # We don't want the cue in this
        visual.cue.hide_nodes()

        cam.load_state(camera_state)

        if show_hud:
            hud.init()
            hud.elements[HUDElement.help_text].help_hint.hide()
            hud.update_cue(shot.cue)
        else:
            hud.destroy()

        # Set quaternions for each ball
        for ball in visual.balls.values():
            ball.set_quats(ball._ball.history_cts)

        frames = int(shot.events[-1].time * fps) + 1

        for frame in range(frames):
            for ball in visual.balls.values():
                ball.set_render_state_from_history(ball._ball.history_cts, frame)

            Global.task_mgr.step()
            exporter.save(self.get_image_array())
