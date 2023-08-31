from abc import abstractmethod

from qwak.model.schema import ModelSchema


class QwakModel:
    """
    Base class for all Qwak based models.
    """

    @abstractmethod
    def build(self):
        """
        Responsible for loading the model. This method is invoked during build time (qwak build command)

        Example usage:

        >>> def build(self):
        >>>     ...
        >>>     train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
        >>>     validate_pool = Pool(X_validation, y_validation, cat_features=categorical_features_indices)
        >>>     self.catboost.fit(train_pool, eval_set=validate_pool)

        :return:
        """
        raise ValueError("Please implement build method")

    @abstractmethod
    def predict(self, df):
        """
        Invoked on every API inference request.
        :param df: the inference vector, as a pandas dataframe

        Example usage:

        >>> def predict(self, df) -> pd.DataFrame:
        >>>     return pd.DataFrame(self.catboost.predict(df), columns=['churn'])

        :return: model output (inference results), as a pandas dataframe
        """
        raise ValueError("Please implement predict method")

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

    def schema(self) -> ModelSchema:
        """
        Specification of the model inputs and outputs. Optional method

        Example usage:

        >>> from qwak.model.schema import ModelSchema, ExplicitFeature, InferenceOutput
        >>>
        >>>
        >>> def schema(self) -> ModelSchema:
        >>>     model_schema = ModelSchema(
        >>>     features=[
        >>>         ExplicitFeature(name="State", type=str),
        >>>     ],
        >>>     predictions=[
        >>>         InferenceOutput(name="score", type=float)
        >>>     ])
        >>>     return model_schema

        :return: a model schema specification
        """
        pass

    def _add_mixin(self, mixin):
        """
        Adds a mixin to the model. This is used to add functionality to the model without modifying the model itself.
        For Qwak internal use only. Anything added here might be overwritten during the build process.
        :param mixin: The mixin to add to the model
        :return: None
        """
        self.__class__ = type(self.__class__.__name__, (mixin, self.__class__), {})
