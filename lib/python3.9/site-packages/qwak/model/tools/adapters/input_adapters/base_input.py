from typing import Any


class BaseInputAdapter:
    """
    BaseInputAdapter is an abstraction layer between user defined API callback function
    and prediction request input in a variety of different forms.
    """

    def __init__(self, **kwargs):
        """
        BaseInputAdapter constructor - accept all keyword arguments as configuration (only in runtime)
        :param kwargs:
        """
        pass

    def extract_user_func_args(self, data: Any) -> Any:
        """
        Extract args that user API function is expecting from Inference
        """
        raise NotImplementedError()
