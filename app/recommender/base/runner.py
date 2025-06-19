import os

import pandas as pd

from abc import ABC, abstractmethod


class BaseRunner(ABC):
    """ABC for running various data processing or model training pipelines.

    Attributes:
        project_dir (str):
            The root directory of the project.
    """

    def __init__(self, project_dir: str) -> None:
        """Initializes the BaseRunner with the project directory.

        Args:
            project_dir (str):
                The root directory of the project.

        Returns:
            None
        """

        self.project_dir = project_dir

    @staticmethod
    def _get_file_path(directory: str, file_name: str) -> str:
        """Constructs the full path for a given file name.

        Args:
            directory (str):
                The directory in which the file is located or should be created.
            file_name (str):
                The name of the file.

        Returns:
            str:
                The full path to the file.
        """

        return os.path.join(directory, file_name)

    @staticmethod
    @abstractmethod
    def _filter_columns_for_export(movies_df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def _verify_dataset_files_exist(self) -> None:
        pass

    @abstractmethod
    def _save_output(self, output_data, file_name: str) -> None:
        pass
