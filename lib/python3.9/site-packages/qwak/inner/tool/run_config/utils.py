import functools
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ConfigCliMap:
    """Mapping between cli argument to properties argument.

    Attributes:
        key (str): key of the argument in the cli.
        prop (str): property path as nested json object (obj.prop1.prop2)
        validation_func (Callable): function which accept two parameters value (value to validate and is_required param.
        is_required (bool): True if value is required else False.
    """

    key: str
    prop: str
    validation_func: Callable = field(default=lambda: True)
    is_required: bool = False


def validate_list_of_strings(value: Any, is_required: bool) -> bool:
    """Validate if a list include only strings

    Args:
        value: Value to perform validation on.
        is_required: True if value is required.

    Returns:
        bool: True if value is valid.

    Examples:
        >>> validate_bool(["1", "3"], False)
        True
        >>> validate_bool(["1", 3], False)
        False
        >>> validate_bool(None, True)
        False
    """
    if not value and not is_required:
        return True
    elif not isinstance(value, Iterable):
        return False

    return all(isinstance(p, str) for p in value)


def validate_string(value: Any, is_required: bool) -> bool:
    """Validate if value is a string.

    Args:
        value: Value to perform validation on.
        is_required: True if value is required.

    Returns:
        bool: True if value is valid.

    Examples:
        >>> validate_string("3", False)
        True
        >>> validate_string(3, False)
        False
        >>> validate_string(None, True)
        False
    """
    if value:
        return isinstance(value, str)
    elif is_required:
        return False

    return True


def validate_float(value: Any, is_required: bool) -> bool:
    """Validate if value is a float.

    Args:
        value: Value to perform validation on.
        is_required: True if value is required.

    Returns:
        bool: True if value is valid.

    Examples:
        >>> validate_float(3, True)
        True
        >>> validate_float(3.5, True)
        True
        >>> validate_float(None, True)
        False
    """
    if value:
        return isinstance(value, (float, int))
    elif is_required:
        return False

    return True


def validate_int(value: Any, is_required: bool) -> bool:
    """Validate if value is a int.

    Args:
        value: Value to perform validation on.
        is_required: True if value is required.

    Returns:
        bool: True if value is valid.

    Examples:
        >>> validate_int(3, True)
        True
        >>> validate_int(3.5, True)
        False
        >>> validate_int(None, True)
        False
    """
    if value:
        return isinstance(value, int) or (isinstance(value, str) and value.isdigit())
    elif is_required:
        return False

    return True


def validate_bool(value: Any, is_required: bool) -> bool:
    """Validate if value is a boolean.

    Args:
        value: Value to perform validation on.
        is_required: True if value is required.

    Returns:
        bool: True if value is valid.

    Examples:
        >>> validate_bool(True, False)
        True
        >>> validate_bool("true", False)
        False
        >>> validate_bool(None, True)
        False
    """
    if value is not None:
        return isinstance(value, bool)
    elif is_required:
        return False

    return True


def validate_enum(enum: Any) -> Callable:
    """Create function to validate if value in enum.

    Args:
        enum: enum to validate on

    Returns:
        bool: Callable.
    """

    def validate(value: Any, is_required: bool) -> bool:
        if value and isinstance(value, str):
            return value.upper() in enum.__members__
        elif is_required:
            return False

        return True

    return validate


def validate_variation(value: Any) -> bool:
    """
    Args:
        value: alue to perform validation on.

    Returns:
        bool: True if value is valid.
    """
    if not (hasattr(value, "name") and isinstance(value.name, str)):
        return False

    if not hasattr(value, "traffic"):
        return False

    if not (
        hasattr(value.traffic, "percentage")
        and isinstance(value.traffic.percentage, int)
    ):
        return False

    if not (
        hasattr(value.traffic, "shadow") and isinstance(value.traffic.shadow, bool)
    ):
        return False

    return True


def validate_variations(value: Any, is_required: bool) -> bool:
    """Validate if value is a variation.

    Args:
        value: Value to perform validation on.

    Returns:
        bool: True if value is valid.

    Examples:
        >>> validate_variations([DeployConfig.Realtime.VariationConfig(name='default', traffic=DeployConfig.Realtime.VariationConfig.TrafficConfig(percentage=100, shadow=False))], False)
        True
        >>> validate_variations(["{variation.name: Test}"], False)
        False
        >>> validate_variations(None, True)
        False
    """
    if not value and not is_required:
        return True
    elif not isinstance(value, Iterable):
        return False

    return all(validate_variation(variation) for variation in value)


def rsetattr(obj: object, attr: str, val: Any):
    """Set attribute in object recursively.

    Args:
        obj: Object to set value in.
        attr: Nested attributes path to value. (obj.prop1.prop2)
        val: Value to set in property
    """
    pre, _, post = attr.rpartition(".")
    setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj: object, attr: str, *args) -> Any:
    """Get attribute in object recursively.

    Args:
        obj: Object to set value in.
        attr: Nested attributes path to value. (obj.prop1.prop2)
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))
