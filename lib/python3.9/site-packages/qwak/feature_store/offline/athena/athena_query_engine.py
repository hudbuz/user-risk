import time
import uuid

import pandas as pd
from _qwak_proto.qwak.ecosystem.v0.ecosystem_runtime_service_pb2 import (
    GetCloudCredentialsParameters,
    GetCloudCredentialsRequest,
    OfflineFeatureStoreClient,
    PermissionSet,
)
from google.protobuf.duration_pb2 import Duration
from qwak.clients.administration.eco_system.client import EcosystemClient
from qwak.exceptions import QwakException
from qwak.feature_store.offline._query_engine import BaseQueryEngine

RECONNECT_THRESHOLD_SEC = 300


class AthenaQueryEngine(BaseQueryEngine):
    def __init__(self):
        eco_client = EcosystemClient()
        self.bucket, environment_id = self._get_env_details(eco_client)

        self.staging_folder_prefix = (
            f"{environment_id}/tmp/offline_fs/{str(uuid.uuid4())}"  # nosec B108
        )
        self.temp_join_table_base_folder = (
            f"s3://{self.bucket}/{self.staging_folder_prefix}"
        )

        self.conn, self.expiration_time = self._init_connection()
        self.cursor = self.conn.cursor()

        self.join_table_specs = []

        self.join_tables_db_name = f"qwak_temp_data_{environment_id.replace('-', '_')}"
        self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.join_tables_db_name}")

    @staticmethod
    def _get_env_details(eco_client):
        environment_configuration = eco_client.get_environment_configuration()

        return (
            environment_configuration.configuration.object_storage_bucket,
            environment_configuration.id,
        )

    def _init_connection(self):
        try:
            # obtain credentials through STS
            eco_client = EcosystemClient()
            cloud_credentials_response = eco_client.get_cloud_credentials(
                request=GetCloudCredentialsRequest(
                    parameters=GetCloudCredentialsParameters(
                        duration=Duration(seconds=60 * 60, nanos=0),
                        permission_set=PermissionSet(
                            offline_feature_store_client=OfflineFeatureStoreClient()
                        ),
                    )
                )
            )

            aws_credentials = (
                cloud_credentials_response.cloud_credentials.aws_temporary_credentials
            )

            try:
                from pyathena import connect
                from pyathena.pandas.cursor import PandasCursor
            except ImportError:
                raise QwakException(
                    """
                    Missing 'pyathena' dependency required for fetching data from the offline store.
                    Please pip install pyathena
                """
                )

            conn = connect(
                s3_staging_dir=self.temp_join_table_base_folder,
                aws_access_key_id=aws_credentials.access_key_id,
                aws_secret_access_key=aws_credentials.secret_access_key,
                aws_session_token=aws_credentials.session_token,
                region_name=aws_credentials.region,
                cursor_class=PandasCursor,
            )

            return (
                conn,
                aws_credentials.expiration_time.seconds,
            )

        except QwakException as e:
            raise e

        except Exception as e:
            raise QwakException(
                f"Got an error trying to retrieve credentials to query the offline store "
                f"in the cloud, error is: {e}"
            )

    def upload_table(self, df: pd.DataFrame):
        join_table_spec = super().JoinTableSpec(
            self.join_tables_db_name, AthenaQueryEngine.get_quotes()
        )
        self.join_table_specs.append(join_table_spec)

        from pyathena.pandas.util import to_sql

        to_sql(
            df,
            join_table_spec.table_name,
            self.conn,
            f"{self.temp_join_table_base_folder}/{join_table_spec.table_name}/",
            schema=self.join_tables_db_name,
            index=False,
            if_exists="replace",
        )

        return join_table_spec.join_table_full_path

    def run_query(self, query: str):
        self._check_reconnection()
        return self.cursor.execute(query).fetchall()

    def read_pandas_from_query(self, query: str, parse_dates=None):
        self._check_reconnection()
        return pd.read_sql(
            query,
            self.conn,
            parse_dates=parse_dates,
        )

    def _check_reconnection(self):
        if self.expiration_time - time.time() < RECONNECT_THRESHOLD_SEC:
            self.conn, self.expiration_time = self._init_connection()
            self.cursor = self.conn.cursor()

    def cleanup(self):
        self._check_reconnection()
        for join_table_spec in self.join_table_specs:
            self.cursor.execute(
                f"""DROP TABLE {join_table_spec.join_table_full_path.replace('"', '`')}"""
            )

        self.join_table_specs = []

        s3 = self.conn.session.resource("s3")
        bucket = s3.Bucket(self.bucket)
        bucket.objects.filter(Prefix=self.staging_folder_prefix).delete()

    @staticmethod
    def get_quotes():
        return '"'
