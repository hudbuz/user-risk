import shutil
from pathlib import Path


def zip_logs(log_path: str, build_id: str):
    if build_id:
        shutil.make_archive(
            base_name=str(Path(log_path).parent / build_id),
            format="zip",
            root_dir=log_path,
        )
        shutil.rmtree(log_path, ignore_errors=True)
