from pathlib import Path

from app.scripts import (
    run_dataset_processer,
    run_feature_engineer,
    run_movie_recommender_trainer,
    run_movie_recommender_engine,
)


SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "app" / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_DIR / "app" / "models"


def main() -> None:
    str_project_dir = str(PROJECT_DIR)

    run_dataset_processer(str_project_dir, str(RAW_DATA_DIR))
    run_feature_engineer(str_project_dir, str(INTERIM_DATA_DIR))
    run_movie_recommender_trainer(str_project_dir, str(PROCESSED_DATA_DIR))
    run_movie_recommender_engine(str_project_dir, str(MODELS_DIR), 296)


if __name__ == "__main__":
    main()
