from recommender.dataset_processer import DatasetProcesser


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
