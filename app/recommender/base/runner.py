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
    def _join_paths(base_path: str, *path_segments: str) -> str:
        """Joins multiple path segments into a single, complete path.

        Args:
            base_path (str):
                The initial path component (e.g., a directory, a root path).
            *path_segments (str):
                One or more additional path components to be joined to the base_path.
                These can be subdirectories, file names, or other path parts.

        Returns:
            str:
                The full, combined path.
        """

        return os.path.join(base_path, *path_segments)

    @abstractmethod
    def run(self) -> None:
        pass

    def _verify_required_files_exist(
        self, directory: str, required_files: dict[str, str]
    ) -> None:
        """Check the existence of all specified required files within a given directory.

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
                If one or more of the required files
                do not exist in the specified directory.
        """

        missing_files = [
            expected_name
            for expected_name, actual_name in required_files.items()
            if not os.path.exists(self._join_paths(directory, actual_name))
        ]

        if missing_files:
            raise FileNotFoundError(
                f"The following required files are missing: {', '.join(missing_files)}"
            )
