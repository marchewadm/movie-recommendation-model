from pathlib import Path

from scripts import run_dataset_processer


PROJECT_DIR = Path.cwd()
RAW_DATA_DIR = PROJECT_DIR / "app" / "data" / "raw"


def main() -> None:
    run_dataset_processer(str(PROJECT_DIR), str(RAW_DATA_DIR))


if __name__ == "__main__":
    main()
