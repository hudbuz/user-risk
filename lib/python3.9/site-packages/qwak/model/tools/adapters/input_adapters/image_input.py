import os
from typing import BinaryIO

from qwak.exceptions import QwakHTTPException

from .file_input import FileInput


class ImageInput(FileInput):
    def __init__(
        self,
        **base_kwargs,
    ):
        try:
            import imageio  # noqa: F401
        except ImportError:
            raise ImportError("imageio package is required to use ImageInput adapter")

        super().__init__(**base_kwargs)
        self.accept_image_formats = {".jpg", ".png", ".jpeg", ".tiff", ".webp", ".bmp"}

    def extract_user_func_args(self, data: BinaryIO) -> "numpy.ndarray":  # noqa F821
        try:
            import imageio
        except ImportError:
            raise ImportError("imageio package is required to use ImageInput adapter")

        if getattr(data, "name", None) and not self.check_file_extension(
            data.name, self.accept_image_formats
        ):
            raise QwakHTTPException(
                400,
                message=f"Current service only accepts {self.accept_image_formats} formats",
            )
        try:
            return imageio.imread(data, pilmode="RGB")
        except ValueError as e:
            raise QwakHTTPException(400, message=str(e))

    @staticmethod
    def check_file_extension(file_name, supported_types):
        if not file_name:
            return False
        _, extension = os.path.splitext(file_name)
        return extension.lower() in (supported_types or [])
