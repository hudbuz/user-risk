from collections import defaultdict
from datetime import date, datetime
from functools import reduce
from typing import DefaultDict, Dict, List, Tuple, Union

from qwak.clients.administration.eco_system.client import EcosystemClient

try:
    import pandas as pd
except ImportError:
    pass

from dateutil.parser import ParserError
from qwak.exceptions import QwakException
from qwak.feature_store._common.functions import normalize_cols
from qwak.feature_store.offline.athena.athena_query_engine import AthenaQueryEngine


class OfflineClient:
    """
    A class used to retrieve data from the offline store - mainly used to get train data for models.
    It requires qwak configure and aws access.
    """

    # Default SCD parameters of the feature store etl
    ANALYTICS_DB_PREFIX = "qwak_analytics_feature_store"
    FS_DB_PREFIX = "qwak_feature_store"
    FS_TABLE_NAME_PREFIX = "offline_feature_store"
    FS_START_TIME_COLUMN = "start_timestamp"
    FS_END_TIME_COLUMN = "end_timestamp"
    FEATURE_STORE_PREFIX = "feature_store"

    DEFAULT_NUMBER_OF_SAMPLE_DATA_ROWS = "100"

    def __init__(
        self,
        query_engine=None,
        environment_id=None,
    ):
        self.query_engine = (
            query_engine if query_engine is not None else AthenaQueryEngine()
        )
        self.quotes = self.query_engine.get_quotes()
        if environment_id is None:
            user_context = EcosystemClient().get_authenticated_user_context().user
            environment_id = (
                user_context.account_details.default_environment_id.replace("-", "_")
            )

        self.environment_id = environment_id.replace("-", "_")
        self.FS_DB_NAME = self.FS_DB_PREFIX + "_" + self.environment_id
        self.FS_ANALYTICS_DB_NAME = self.ANALYTICS_DB_PREFIX + "_" + self.environment_id

    def get_feature_range_values(
        self,
        entity_key_to_features: dict,
        start_date: Union[datetime, date],
        end_date: Union[datetime, date],
    ):
        """
        :param entity_key_to_features: a dictionary { entity_key(s) -> features list }.
        :param start_date: the column name of the point in time column (default - timestamp)
        :param end_date: the column name of the point in time column (default - timestamp)
        :return: a pandas dataframe or a list of dataframes (a dataframe for every entity_key) - all feature values for
         all entites under the given date range

        each row in the returned data-frame is constructed by retrieving the requested features of the entity
         key(s) for all entity values in within the defined date tange.

        Feature sets should be named [Feature Set Name].[Feature Name],
        i.e: user_purchases.number_of_purchases.

        Examples:
        >>> from datetime import datetime
        >>> from qwak.feature_store.offline import OfflineClient
        >>>
        >>> start_date = datetime(year=2021, month=1, day=1)
        >>> end_date = datetime(year=2021, month=1, day=3)
        >>>
        >>> key_to_features = {'uuid': ['user_purchases.number_of_purchases',
        >>>                             'user_purchases.avg_purchase_amount']}
        >>>
        >>> offline_feature_store = OfflineClient()
        >>>
        >>> train_df = offline_feature_store.get_feature_range_values(
        >>>                entity_key_to_features=key_to_features,
        >>>                start_date=start_date,
        >>>                end_date=end_date)
        >>>
        >>> print(train_df.head())
        >>> #	     uuid	         timestamp	      user_purchases.number_of_purchases	user_purchases.avg_purchase_amount
        >>> # 0	      1	        2021-01-02 17:00:00	                 76	                                4.796842
        >>> # 1	      1	        2021-01-01 12:00:00	                 5	                                1.548000
        >>> # 2	      2	        2021-01-02 12:00:00	                 5	                                5.548000
        >>> # 3	      2	        2021-01-01 18:00:00	                 5	                                2.788000
        """
        try:
            from qwak.feature_store._common.featureset_asterisk_handler import (
                unpack_asterisk_features_from_key_mapping,
            )

            entity_key_to_features = unpack_asterisk_features_from_key_mapping(
                entity_key_to_features, lambda: self
            )
            self._validate_range_query_inputs(
                entity_key_to_features, start_date, end_date
            )

            feature_set_name_to_feature_list = (
                self._partition_feature_set_by_entity_feature(entity_key_to_features)
            )
            feature_set_to_dtypes = self._validate_database_and_get_feature_set_dtypes(
                feature_set_name_to_feature_list
            )
            self._validate_features_exist(
                feature_set_name_to_feature_list, feature_set_to_dtypes
            )

            df = self._run_ranges_query(
                feature_set_name_to_feature_list, start_date, end_date
            )

            return self._normalize_df(df)

        except QwakException as qwak_exception:
            raise QwakException(
                f"Got the following Qwak generated exception: {qwak_exception}"
            )
        except Exception as e:
            raise QwakException(f"Got the following run-time exception: {e}")
        finally:
            try:
                self.query_engine.cleanup()
            except Exception as e:
                print(f"Got the following run-time exception during cleanup: {e}")

    def get_feature_values(
        self,
        entity_key_to_features: dict,
        population: "pd.DataFrame",
        point_in_time_column_name: str = "timestamp",
    ):
        """
        :param entity_key_to_features: a dictionary { entity_key(s) -> features list }.
        :param population: a pandas data-frame with a point in time column
                           and a column for each entity key defined at entity_key_to_features.
        :param point_in_time_column_name: the column name of the point in time column (default - timestamp)
        :return: a pandas data-frame - the population joined with the feature values for all
                                       the requested entities and features.

        each row in the returned data-frame is constructed by retrieving the requested features of the entity key(s) for
        the specific entity value(s) in the population and on the specific point in time defined.

        Feature sets should be named [Feature Set Name].[Feature Name],
        i.e: user_purchases.number_of_purchases.

        Examples:
        >>> import pandas as pd
        >>> from qwak.feature_store.offline import OfflineClient
        >>>
        >>> population_df = pd.DataFrame(
        >>>     columns= ['uuid',       'timestamp'     ],
        >>>     data   =[[ '1'  , '2021-01-02 17:00:00' ],
        >>>              [ '2'  , '2021-01-01 12:00:00' ]])
        >>>
        >>> key_to_features = {'uuid': ['user_purchases.number_of_purchases',
        >>>                             'user_purchases.avg_purchase_amount']}
        >>>
        >>> offline_feature_store = OfflineClient()
        >>>
        >>> train_df = offline_feature_store.get_feature_values(
        >>>                entity_key_to_features=key_to_features,
        >>>                population=population_df,
        >>>                point_in_time_column_name='timestamp')
        >>>
        >>> print(train_df.head())
        >>> #	     uuid	         timestamp	      user_purchases.number_of_purchases	user_purchases.avg_purchase_amount
        >>> # 0	      1	        2021-04-24 17:00:00	                 76	                                4.796842
        >>> # 1	      2	        2021-04-24 12:00:00	                 5	                                1.548000
        """

        import pandas as pd

        try:
            from qwak.feature_store._common.featureset_asterisk_handler import (
                unpack_asterisk_features_from_key_mapping,
            )

            population = population.copy()

            entity_key_to_features = unpack_asterisk_features_from_key_mapping(
                entity_key_to_features, lambda: self
            )

            self._validate_point_in_time_query_inputs(
                entity_key_to_features, population, point_in_time_column_name
            )

            feature_set_name_to_feature_dict = (
                self._partition_feature_set_by_entity_feature(entity_key_to_features)
            )

            feature_set_to_dtypes = self._validate_database_and_get_feature_set_dtypes(
                feature_set_name_to_feature_dict
            )

            self._validate_features_exist(
                feature_set_name_to_feature_dict,
                feature_set_to_dtypes,
            )

            population = self._align_entity_key_dtype(
                feature_set_to_dtypes, entity_key_to_features, population
            )

            uploaded_population_path = self.query_engine.upload_table(population)

            df = pd.DataFrame()

            if feature_set_name_to_feature_dict:
                df = self._run_point_in_time_query(
                    feature_set_name_to_feature_dict,
                    uploaded_population_path,
                    point_in_time_column_name,
                    [column.lower() for column in population.columns],
                )

            return self._normalize_df(df)

        except QwakException as qwak_exception:
            raise QwakException(
                f"Got the following Qwak generated exception: {qwak_exception}"
            )
        except Exception as e:
            raise QwakException(f"Got the following run-time exception: {e}")
        finally:
            try:
                self.query_engine.cleanup()
            except Exception as e:
                print(f"Got the following run-time exception during cleanup: {e}")

    @staticmethod
    def _normalize_df(df: "pd.DataFrame") -> "pd.DataFrame":
        columns = df.columns.values.tolist()
        new_columns = normalize_cols(columns)
        df.columns = new_columns
        return df

    @staticmethod
    def _validate_range_query_inputs(
        entity_key_to_features: dict, start_date: datetime, end_date: datetime
    ):
        missing_features_entity_keys = [
            entity_key
            for entity_key, features in entity_key_to_features.items()
            if not features
        ]

        if missing_features_entity_keys:
            raise QwakException(
                f"Features of an entity key must exist, missing features for: [{missing_features_entity_keys}]"
            )

        if (end_date - start_date).total_seconds() < 0:
            raise QwakException("Invalid date range - end date is before start date")

    @staticmethod
    def _validate_point_in_time_query_inputs(
        entity_key_to_features: dict,
        population: "pd.DataFrame",
        point_in_time_column_name: str,
    ):
        """
        Validates that the entity keys, timestamp cols and features exist in DB
        """
        missing_keys = [
            entity_key
            for entity_key in entity_key_to_features.keys()
            if entity_key not in population
        ]
        if missing_keys:
            raise QwakException(
                f"The entity keys must be in population_df columns, missing: [{missing_keys}]"
            )

        missing_features_entity_keys = [
            entity_key
            for entity_key, features in entity_key_to_features.items()
            if not features
        ]

        if missing_features_entity_keys:
            raise QwakException(
                f"Features of an entity key must exist, missing features for: [{missing_features_entity_keys}]"
            )

        if point_in_time_column_name not in population:
            raise QwakException(
                "The point in time column must be part of the population dataframe"
            )

        from pandas.api.types import is_datetime64_any_dtype

        if not is_datetime64_any_dtype(population[point_in_time_column_name]):
            try:
                population[point_in_time_column_name] = pd.to_datetime(
                    population[point_in_time_column_name]
                )
            except ParserError as e:
                raise QwakException(
                    f"It was not possible to cast provided point in time column to datetime"
                    f"\nError message: {e}"
                )

    @staticmethod
    def _partition_feature_set_by_entity_feature(
        entity_key_to_features,
    ) -> DefaultDict[Tuple[str, str], List[str]]:
        """
        Partition feature by entity key and featureset name
        Args:
            entity_key_to_features: dict of entity_key -> full feature name
        Returns:
            dict of (entity_key,featureset_name) -> list of feature names
        """
        feature_name_to_feature_list = defaultdict(list)

        for entity_key, feature_list in entity_key_to_features.items():
            for feature in feature_list:
                split_feature_set_and_feature = feature.lower().split(".")
                if len(split_feature_set_and_feature) != 2:
                    raise QwakException(
                        f"Failed to verify features. Name should be: <feature set name>.<feature name>. "
                        f"Current name is: {feature}"
                    )
                feature_set_name = split_feature_set_and_feature[0]
                feature_name_to_feature_list[(entity_key, feature_set_name)].append(
                    feature
                )

        return feature_name_to_feature_list

    def _validate_database_and_get_feature_set_dtypes(
        self, feature_name_to_feature_list: DefaultDict[Tuple[str, str], List[str]]
    ) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        """
        Args:
            feature_name_to_feature_list:  dictionary from feature set name to its' features

        Returns
            dictionary from feature set name and entity key to a list of feature name, feature type
        """
        if self.FS_DB_NAME not in self._fs_db_names():
            raise QwakException("Offline feature store does not contain any data")

        feature_set_to_dtypes = {}
        for (
            entity_key,
            feature_set_name,
        ), feature_list in feature_name_to_feature_list.items():
            table_name = self._get_offline_feature_store_full_name(feature_set_name)
            if table_name not in self._fs_tables_names():
                raise QwakException(
                    f"[{feature_set_name}] feature set does not contain any data"
                )

            columns_query_result = self.query_engine.run_query(
                f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS "  # nosec B608
                f"WHERE TABLE_SCHEMA = '{self.FS_DB_NAME}' "
                f"AND TABLE_NAME = '{table_name}'"
            )

            feature_set_to_dtypes[(entity_key, feature_set_name)] = [
                (column_tup[3], column_tup[7]) for column_tup in columns_query_result
            ]
        return feature_set_to_dtypes

    @staticmethod
    def _validate_features_exist(
        feature_name_to_feature_list: defaultdict,
        feature_set_to_dtypes: Dict[Tuple[str, str], List[Tuple[str, str]]],
    ):
        """
        Args:
            feature_name_to_feature_list: dictionary from feature set name to its' features
            feature_set_to_dtypes: dictionary from feature set name and entity key
                                   to a list of feature name, feature type
        """
        for (
            entity_key,
            feature_set_name,
        ), feature_list in feature_name_to_feature_list.items():
            columns = [
                column_tuple[0].lower()
                for column_tuple in feature_set_to_dtypes[
                    (entity_key, feature_set_name)
                ]
            ]
            absent_features = [
                feature for feature in feature_list if feature not in columns
            ]
            if absent_features:
                raise QwakException(
                    f"Missing the following features for the feature set "
                    f"[{feature_set_name}]:"
                    f"\n{absent_features}"
                )

    def _align_entity_key_dtype(
        self,
        feature_set_to_dtypes: Dict[Tuple[str, str], List[Tuple[str, str]]],
        entity_key_to_features: Dict[str, List[str]],
        population: "pd.DataFrame",
    ) -> "pd.DataFrame":
        """
        Args:
            feature_set_to_dtypes: dictionary from feature set name and entity key
                                   to a list of feature name, feature type
            entity_key_to_features: a dictionary { entity_key(s) -> features list }.
            population: a pandas data-frame with a point in time column
                           and a column for each entity key defined at entity_key_to_features.
        Returns:
            entity type aligned population df
        """
        sql_type_to_pandas = {
            "string": "string",
            "integer": "int",
            "varchar": "string",
            "text": "string",
            "bigint": "int",
        }

        entity_key_to_dtype = self._validate_and_get_entity_keys_dtypes(
            entity_key_to_features, population
        )

        for (
            entity_key,
            feature_set_name,
        ), feature_dtypes_list in feature_set_to_dtypes.items():
            given_entity_key_dtype = entity_key_to_dtype[entity_key]
            entity_column_tuple = [
                column_tuple
                for column_tuple in feature_dtypes_list
                if column_tuple[0] == entity_key
            ]
            if not entity_column_tuple:
                raise QwakException(
                    f"Did not find entity key [{entity_key}] in the table of [{feature_set_name}] "
                    f"- existing columns are: "
                    f"{[column_tuple[0] for column_tuple in feature_dtypes_list]}"
                )
            actual_entity_type = entity_column_tuple[0][1]
            if actual_entity_type == given_entity_key_dtype:
                continue
            else:
                try:
                    population[entity_key] = population[entity_key].astype(
                        sql_type_to_pandas[actual_entity_type]
                    )
                    print(
                        f"Entity [{entity_key}] given type [{given_entity_key_dtype}] "
                        f"was not aligned with actual type [{actual_entity_type}] - casted to correct type"
                    )
                except ValueError as e:
                    raise QwakException(
                        f"Mismatched entity type for [{entity_key}] - [{given_entity_key_dtype}] "
                        f"- failed to cast to actual type [{actual_entity_type}], Error: {e}"
                    )

        return population

    def _validate_and_get_entity_keys_dtypes(
        self,
        entity_key_to_features: Dict[str, List[str]],
        population_df: "pd.DataFrame",
    ) -> Dict[str, str]:
        """
        Args:
            entity_key_to_features: a dictionary { entity_key(s) -> features list }.
            population_df: a pandas data-frame with a point in time column
                           and a column for each entity key defined at entity_key_to_features.

        Returns:
            dictionary of entity key to it's dtype
        """
        supported_dtypes_to_actual_type = {
            "object": "string",
            "int32": "integer",
            "int64": "integer",
        }
        entity_key_to_dtype = {}
        for entity_key in entity_key_to_features.keys():
            entity_pandas_dtype = population_df.dtypes[entity_key].name
            if entity_pandas_dtype not in supported_dtypes_to_actual_type:
                raise QwakException(
                    f"Got an unsupported dtype for the entity key "
                    f"[{entity_key}] - [{entity_pandas_dtype}]"
                )
            entity_key_to_dtype[entity_key] = supported_dtypes_to_actual_type[
                entity_pandas_dtype
            ]
        return entity_key_to_dtype

    def _run_ranges_query(
        self,
        feature_name_to_feature_list: defaultdict,
        start_date: datetime,
        end_date: datetime,
    ):
        result_dfs = []
        features_set_by_entity = defaultdict(lambda: defaultdict(set))
        for (
            (entity_key, feature_set_name),
            feature_list,
        ) in feature_name_to_feature_list.items():
            for feature in feature_list:
                feature_set_name = feature.split(".")[0]
                features_set_by_entity[entity_key][feature_set_name].add(feature)

        for entity_key, features_dict in features_set_by_entity.items():
            entity_dfs = []
            for feature_set_name, feature_list in features_dict.items():
                offline_feature_store_full_path, table_path = self.get_fs_full_path(
                    feature_set_name
                )

                features = ", ".join(
                    [
                        f"{offline_feature_store_full_path}.{self.quotes}{feature}{self.quotes}"
                        for feature in feature_list
                    ]
                )

                where_part = (
                    "WHERE "
                    f"{table_path}.{self.FS_START_TIME_COLUMN} >= timestamp '{start_date}' "
                    f"AND ({self.FS_END_TIME_COLUMN} <= "
                    f"timestamp '{end_date}' OR {table_path}.{self.FS_END_TIME_COLUMN} IS NULL) "
                    f"AND {table_path}.{self.FS_START_TIME_COLUMN} < timestamp '{end_date}'"
                )

                full_sql = (
                    f"SELECT {offline_feature_store_full_path}.{self.quotes}{entity_key}{self.quotes}, "  # nosec B608
                    f"{offline_feature_store_full_path}.{self.quotes}{self.FS_START_TIME_COLUMN}{self.quotes}, "
                    f"{features} "
                    f"FROM {offline_feature_store_full_path} "
                    f"{where_part}"
                )

                entity_dfs.append(self.query_engine.read_pandas_from_query(full_sql))

            entity_final_df = reduce(
                lambda left, right: pd.merge(
                    left, right, on=[entity_key, self.FS_START_TIME_COLUMN], how="outer"
                ),
                entity_dfs,
            )
            result_dfs.append(entity_final_df.reset_index(drop=True))

        return result_dfs[0] if len(result_dfs) == 1 else result_dfs

    def _run_point_in_time_query(
        self,
        feature_name_to_feature_list: defaultdict,
        uploaded_population_path: str,
        point_in_time_column_name: str,
        population_list: list,
    ) -> "pd.DataFrame":
        """
        creates SQL query for pint in time feature fetching based on population and requested features
        """
        dfs = []

        for index, ((entity_key, feature_set_name), feature_list) in enumerate(
            feature_name_to_feature_list.items()
        ):
            offline_feature_store_full_path, table_path = self.get_fs_full_path(
                feature_set_name
            )

            join_part = self._get_join_population_sql(
                entity_key, offline_feature_store_full_path, uploaded_population_path
            )

            point_in_time_column_full_path = (
                f"{uploaded_population_path}.{point_in_time_column_name}"
            )

            where_part = (
                "WHERE "
                f"{point_in_time_column_full_path} >= "
                f"{table_path}.{self.FS_START_TIME_COLUMN} "
                f"AND ({point_in_time_column_full_path} < "
                f"{table_path}.{self.FS_END_TIME_COLUMN} OR "
                f"{table_path}.{self.FS_END_TIME_COLUMN} IS NULL)"
            )

            features = ", ".join(
                [
                    f"{offline_feature_store_full_path}.{self.quotes}{feature}{self.quotes} as {self.quotes}{feature}{self.quotes}"
                    for feature in feature_list
                ]
            )

            final_query_features = ", ".join(
                [
                    f"filtered_features.{self.quotes}{feature}{self.quotes} as {self.quotes}{feature}{self.quotes}"
                    for feature in feature_list
                ]
            )

            full_sql = (
                f"WITH "  # nosec B608
                "filtered_features AS ( "
                f"SELECT {uploaded_population_path}.*, {features} "
                f"FROM {uploaded_population_path} "
                f"{join_part} "
                f"{where_part} "
                ") "
                f"SELECT population.*, {final_query_features} "
                f"FROM {uploaded_population_path} population "
                f"LEFT JOIN filtered_features "
                f"ON population.{entity_key} = filtered_features.{entity_key} "
                f"AND population.{point_in_time_column_name} = filtered_features.{point_in_time_column_name}"
            )

            dfs.append(
                self.query_engine.read_pandas_from_query(
                    full_sql, [point_in_time_column_name]
                )
            )
        return self._merge_query_dataframes_results(dfs, population_list)

    @staticmethod
    def _merge_query_dataframes_results(
        dfs: List["pd.DataFrame"], population_list
    ) -> "pd.DataFrame":
        """
        merges query result dataframes according to population list cols
        """
        if dfs:
            df_final = reduce(
                lambda left, right: pd.merge(
                    left, right, on=population_list, how="outer"
                ),
                dfs,
            )

            ordered_cols = population_list + (
                df_final.columns.drop(population_list).tolist()
            )

            return df_final[ordered_cols].reset_index(drop=True)
        else:
            return pd.DataFrame()

    def _get_join_population_sql(
        self, entity_key, offline_feature_store_full_path, uploaded_population_path
    ):
        """
        return a join sql query for uploaded population table
        """
        join_part = (
            f"LEFT JOIN {offline_feature_store_full_path} ON "
            f"{offline_feature_store_full_path}.{entity_key} = "
            f"{uploaded_population_path}.{entity_key} "
        )
        return join_part

    def get_fs_full_path(self, feature_set_name: str) -> Tuple[str, str]:
        offline_feature_store_table_name = self._get_offline_feature_store_full_name(
            feature_set_name
        )
        table_path = f"{self.quotes}{offline_feature_store_table_name}{self.quotes}"
        offline_feature_store_full_path = (
            f"{self.quotes}{self.FS_DB_NAME}{self.quotes}."
            f"{self.quotes}{offline_feature_store_table_name}{self.quotes}"
        )
        return offline_feature_store_full_path, table_path

    def _fs_db_names(self) -> List[str]:
        return [
            database_tuple[0]
            for database_tuple in self.query_engine.run_query("SHOW SCHEMAS")
        ]

    def _fs_tables_names(self) -> List[str]:
        return [
            table_tuple[0]
            for table_tuple in self.query_engine.run_query(
                f"SHOW TABLES IN {self.FS_DB_NAME}"
            )
        ]

    def _get_offline_feature_store_full_name(self, feature_set_name: str) -> str:
        return f"{self.FS_TABLE_NAME_PREFIX}_{feature_set_name}".lower().replace(
            "-", "_"
        )
