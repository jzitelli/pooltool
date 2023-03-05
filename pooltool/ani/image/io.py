import re
from pathlib import Path
from typing import List, Union

import attrs
import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from pooltool.ani.image.utils import DataPack, ImageExt, gif


@attrs.define
class ImageDir:
    """Exporter for creating a directory of images"""

    path: Path = attrs.field(converter=Path)
    ext: ImageExt = attrs.field(converter=ImageExt)
    prefix: str = attrs.field(default="shot")
    save_gif: bool = attrs.field(default=False)
    image_count: int = attrs.field(init=False, default=0)
    paths: List[Path] = attrs.field(init=False, factory=list)

    def __attrs_post_init__(self):
        if not self.path.exists():
            self.path.mkdir(parents=True)

    def save(self, data: DataPack) -> None:
        frames = np.shape(data.imgs)[0]
        for frame in range(frames):
            path = self._get_filepath()
            assert not path.exists(), f"{path} already exists!"

            plt.imsave(path, data.imgs[frame, :, :, :])

            # Increment
            self.image_count += 1
            self.paths.append(path)

        if data.system is not None:
            data.system.save(self.path / f"_{self.prefix}.msgpack")

        if self.save_gif:
            gif(
                paths=self.paths,
                output=self.path / f"_{self.prefix}.gif",
                fps=data.fps,
            )

    def _get_filepath(self) -> Path:
        stem = f"{self.prefix}_{self.image_count:06d}"
        name = f"{stem}.{self.ext}"
        return Path(self.path) / name

    @staticmethod
    def read(path: Union[str, Path]) -> NDArray[np.uint8]:
        path = Path(path)

        assert path.exists(), f"{path} is not a directory"

        img_pattern = re.compile(r".*_[0-9]{6,6}\." + ImageExt.regex())

        return np.array(
            [
                np.asarray(Image.open(img_path))[:, :, :3]
                for img_path in sorted(path.glob("*"))
                if img_pattern.match(str(img_path))
            ],
            dtype=np.uint8,
        )


@attrs.define
class HDF5Images:
    path: Path = attrs.field(converter=Path)

    def save(self, data: DataPack) -> None:
        with h5py.File(self.path, "w") as fp:
            fp.create_dataset(
                "images", np.shape(data.imgs), h5py.h5t.STD_U8BE, data=data.imgs
            )

        if data.system is not None:
            data.system.save(self.path.with_suffix(".msgpack"))

    @staticmethod
    def read(path: Union[str, Path]) -> NDArray[np.uint8]:
        with h5py.File(path, "r+") as fp:
            return np.array(fp["/images"]).astype("uint8")


@attrs.define
class NpyImages:
    path: Path = attrs.field(converter=Path)

    def save(self, data: DataPack) -> None:
        np.save(self.path, data.imgs)
        if data.system is not None:
            data.system.save(self.path.with_suffix(".msgpack"))

    @staticmethod
    def read(path: Union[str, Path]) -> NDArray[np.uint8]:
        return np.load(path)
