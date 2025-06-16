from recommender.dataset_processer import DatasetProcesser
from recommender.modeling.train import MovieRecommenderTrainer


def run_dataset_processer(project_dir: str, raw_data_dir: str) -> None:
    """Initializes and runs the DatasetProcesser.

    Creates an instance of "DatasetProcesser" with the specified
    project and raw data directories, and the names of the required CSV files.
    It then executes the data processing pipeline by calling the "run" method
    of the "DatasetProcesser" instance.

    Args:
        project_dir (str):
            The root directory of the project as a string.
        raw_data_dir (str):
            The directory where the raw MovieLens CSV files are located as a string.

    Returns:
        None
    """

    dataset_processer = DatasetProcesser(
        project_dir,
        raw_data_dir,
        "movies.csv",
        "ratings.csv",
        "links.csv",
        "tags.csv",
    )

    dataset_processer.run()


def run_movie_recommender_trainer(project_dir: str, processed_data_dir: str) -> None:
    """Initializes and runs the MovieRecommenderTrainer.

    This function creates an instance of "MovieRecommenderTrainer" with the specified
    project and processed data directories. It then executes the model training
    pipeline by calling the "run" method of the "MovieRecommenderTrainer" instance.

    Args:
        project_dir (str):
            The root directory of the project as a string.
        processed_data_dir (str):
            The directory where the processed movie data is located as a string.

    Returns:
        None
    """

    movie_recommender_trainer = MovieRecommenderTrainer(
        project_dir,
        processed_data_dir,
        "movies_processed.csv",
    )

    movie_recommender_trainer.run()
