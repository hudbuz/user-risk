import pandas as pd
import qwak
from qwak.model.base import QwakModel


class {{cookiecutter.model_class_name}}(QwakModel):
    """ The Model class inherit QwakModel base class
    """

    def __init__(self):
        pass

    def build(self):
        """ Responsible for loading the model. This method is invoked during build time (qwak build command)

           Example:
           >>> def build(self):
           >>>     ...
           >>>     train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
           >>>     validate_pool = Pool(X_validation, y_validation, cat_features=categorical_features_indices)
           >>>     self.catboost.fit(train_pool, eval_set=validate_pool)
           """
        pass

    def schema(self):
        """ Specification of the model inputs and outputs. Optional method

        Example:
        >>> from qwak.model.schema import ModelSchema, InferenceOutput, ExplicitFeature
        >>>
        >>> def schema(self) -> ModelSchema:
        >>>     model_schema = ModelSchema(
        >>>     inputs=[
        >>>         ExplicitFeature(name="State", type=str),
        >>>     ],
        >>>     outputs=[
        >>>         InferenceOutput(name="score", type=float)
        >>>     ])
        >>>     return model_schema

       Returns: a model schema specification
       """
        pass

    def initialize_model(self):
        """
        Invoked when a model is loaded at serving time. Called once per model instance initialization. Can be used for
        loading and storing values that should only be available in a serving setting or loading pretrained models. Optional method

        Example usage:

        >>> def initialize_model(self):
        >>>     with open('model.pkl', 'rb') as infile:
        >>>         self.model = pickle.load(infile)
        """
        pass

    @qwak.api()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Invoked on every API inference request.
        Args:
            pd (DataFrame): the inference vector, as a pandas dataframe

        Returns: model output (inference results), as a pandas dataframe
        """
        pass
