# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "huggingface-hub",
#     "pillow",
#     "vllm>=0.15.1",
#     "tqdm",
#     "torch",
# ]
# ///
"""Sample HF uv-script used for introspection tests.

This file is NEVER executed by ocrscout.sync — it is parsed via ast.parse only.
"""

import argparse
import os

from datasets import load_dataset  # noqa: F401
from huggingface_hub import snapshot_download  # noqa: F401

PROMPT_TEMPLATES = {
    "ocr": "Extract the text content from this image.",
    "layout-all": "Output layout JSON with bbox + category + text.",
    "layout-only": "Output layout JSON with bbox + category only.",
    "doctags": f"Emit DocTags. {'{...}'}",  # exercises JoinedStr handling
}

DEFAULT_MODEL_NAME = "rednote-hilab/dots.mocr"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dataset")
    parser.add_argument("output_dataset")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--output-column", default="markdown")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    # A non-literal default we should NOT crash on:
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--prompt-mode", default="layout-all")
    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    main()
