"""YAML yapılandırmasından N-BaIoT baseline eğitimi başlatır."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.training.baseline_trainer import run_training  # noqa: E402


DEFAULT_CONFIG_FILE = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "family3_tinyml_mlp_smoke.yaml"
)


def parse_arguments() -> argparse.Namespace:
    """Komut satırı parametrelerini oluşturur."""

    parser = argparse.ArgumentParser(
        description=(
            "N-BaIoT baseline eğitimini YAML yapılandırmasıyla çalıştırır."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help="YAML deney yapılandırması.",
    )

    return parser.parse_args()


def main() -> None:
    """Eğitimi başlatır."""

    args = parse_arguments()

    run_training(
        config_file=args.config,
    )


if __name__ == "__main__":
    main()