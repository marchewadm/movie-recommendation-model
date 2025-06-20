import os

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

    @abstractmethod
    def run(self) -> None:
        pass

    def _verify_required_files_exist(
        self, directory: str, required_files: dict[str, str]
    ) -> None:
        """Verifies the existence of all specified required files within a given directory.

        This method iterates through a dictionary of expected file names and their
        corresponding actual file names (as provided during class instantiation).
        It checks if each file exists in the constructed path.

        Args:
            directory (str):
                The directory path where the files are expected to be found.
            required_files (dict[str, str]):
                A dictionary where keys are the expected descriptive
                names of the files (e.g., "example.txt") and values are the actual
                file names (e.g., "self.example_txt_file") used to construct the path.
                For example, { "example.txt": self.example_txt_file }

        Returns:
            None

        Raises:
            FileNotFoundError:
                If one or more of the required files do not exist in the specified directory.
        """

        missing_files = [
            expected_name
            for expected_name, actual_name in required_files.items()
            if not os.path.exists(self._get_file_path(directory, actual_name))
        ]

        if missing_files:
            raise FileNotFoundError(
                f"The following required files are missing: {', '.join(missing_files)}"
            )
