from pathlib import Path

from scripts import run_dataset_processer, run_movie_recommender_trainer


PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "app" / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def main() -> None:
    run_dataset_processer(str(PROJECT_DIR), str(RAW_DATA_DIR))
    run_movie_recommender_trainer(str(PROJECT_DIR), str(PROCESSED_DATA_DIR))


if __name__ == "__main__":
    main()
