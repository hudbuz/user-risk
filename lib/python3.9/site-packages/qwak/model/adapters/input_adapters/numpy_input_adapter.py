import json
import typing as t

from qwak.exceptions import QwakException

from .base_input_adapter import BaseInputAdapter

try:
    import numpy as np
except ImportError:
    pass
    # Numpy is supported either by supplying it as a dependency (or sub-dependency)
    # in your Python project, or during the runtime


def _is_matched_shape(
    left: t.Optional[t.Tuple[int, ...]],
    right: t.Optional[t.Tuple[int, ...]],
) -> bool:  # pragma: no cover
    if (left is None) or (right is None):
        return False

    if len(left) != len(right):
        return False

    for i, j in zip(left, right):
        if i == -1 or j == -1:
            continue
        if i == j:
            continue
        return False
    return True


class NumpyInputAdapter(BaseInputAdapter):
    def __init__(
        self,
        dtype: t.Optional[t.Union[str]] = None,
        enforce_dtype: bool = False,
        shape: t.Optional[t.Tuple[int, ...]] = None,
        enforce_shape: bool = False,
    ):
        if dtype is not None and not isinstance(dtype, np.dtype):
            # Convert from primitive type or type string, e.g.:
            # np.dtype(float)
            # np.dtype("float64")
            try:
                dtype = np.dtype(dtype)
            except TypeError as e:
                raise QwakException(f'NumpyNdarray: Invalid dtype "{dtype}": {e}')

        self._dtype = dtype
        self._shape = shape
        self._enforce_dtype = enforce_dtype
        self._enforce_shape = enforce_shape

    def _verify_ndarray(
        self,
        obj,
        exception_cls: t.Type[Exception] = QwakException,
    ):
        if self._dtype is not None and self._dtype != obj.dtype:
            # ‘same_kind’ means only safe casts or casts within a kind, like float64
            # to float32, are allowed.
            if np.can_cast(obj.dtype, self._dtype, casting="same_kind"):
                obj = obj.astype(self._dtype, casting="same_kind")  # type: ignore
            else:
                msg = (
                    f'{self.__class__.__name__}: Expecting ndarray of dtype "{self._dtype}", but "{obj.dtype}" was '
                    f"received. "
                )
                if self._enforce_dtype:
                    raise exception_cls(msg)
                else:
                    print(msg)

        if self._shape is not None and not _is_matched_shape(self._shape, obj.shape):
            msg = (
                f'{self.__class__.__name__}: Expecting ndarray of shape "{self._shape}", but "{obj.shape}" was '
                f"received. "
            )
            if self._enforce_shape:
                raise exception_cls(msg)
            try:
                obj = obj.reshape(self._shape)
            except ValueError as e:
                print(f"{msg} Failed to reshape: {e}.")

        return obj

    def extract_user_func_arg(self, data: bytes):
        import numpy as np

        data = json.loads(data)
        try:
            res = np.array(data, dtype=self._dtype)  # type: ignore[arg-type]
        except ValueError:
            res = np.array(data)  # type: ignore[arg-type]
        res = self._verify_ndarray(res, QwakException)
        return [res]
