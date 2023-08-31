from pathlib import Path

from cookiecutter import generate
from qwak.exceptions import QwakException


def initialize_model_structure(uri: str, template: str, logger, **template_args):
    if Path(uri).is_file():
        raise QwakException(
            f"Unable to create the model directory structure - given path {uri} is an existing file"
        )

    if not Path(uri).is_dir():
        logger.info(f"Given path {uri} doesn't exists - creating it")
        Path(uri).mkdir(parents=True, exist_ok=True)

    context = generate.generate_context(
        context_file=(
            Path(__file__).parent / "template" / template / "cookiecutter.json"
        ).resolve(),
        default_context=None,
        extra_context={**template_args},
    )

    output_dir = uri + "/" + context.get("cookiecutter", {}).get("model_directory")
    if Path(output_dir).exists():
        raise QwakException(
            f"Folder with the name {output_dir} is already exists. "
            f"Please change model name or change the working path."
        )
    logger.info("Generating structure...")
    generate.generate_files(
        (Path(__file__).parent / "template" / template).resolve(),
        context,
        output_dir=uri,
        overwrite_if_exists=False,
        skip_if_file_exists=False,
    )

    logger.info(f"Created base model at {Path(output_dir).resolve()}")
