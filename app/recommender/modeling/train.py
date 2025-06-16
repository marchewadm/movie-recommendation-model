import os
import pickle

from typing import TypedDict

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from numpy import ndarray


class TrainedModel(TypedDict):
    """A TypedDict to define the structure of the trained recommendation model.

    Attributes:
        movies_df (pd.DataFrame): The DataFrame containing movie metadata.
        tfidf_vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
        tfidf_matrix (csr_matrix): The TF-IDF feature matrix.
        cosine_sim (ndarray): The cosine similarity matrix.
        movie_id_to_index (pd.Series): A mapping from movie ID to DataFrame index.
    """

    movies_df: pd.DataFrame
    tfidf_vectorizer: TfidfVectorizer
    tfidf_matrix: csr_matrix
    cosine_sim: ndarray
    movie_id_to_index: pd.Series


class MovieRecommenderTrainer:
    """Trains a movie recommendation model based on processed movie data.

    This class handles loading the processed movie dataset, extracting features
    using TF-IDF, calculating cosine similarity between movie profiles,
    and saving the trained model components (TF-IDF vectorizer, cosine similarity
    matrix, and movie metadata) for later use in recommendations.

    Attributes:
        project_dir (str):
            The root directory of the project.
        processed_data_dir (str):
            The directory where the processed data is stored.
        movies_csv_file (str):
            The filename of processed movies csv file.
            Defaults to "movies_processed.csv".
    """

    def __init__(
        self,
        project_dir: str,
        processed_data_dir: str,
        movies_csv_file: str = "movies_processed.csv",
    ) -> None:
        """Initializes the MovieRecommenderTrainer with file paths and directories.

        Args:
            project_dir (str):
                The root directory of the project.
            processed_data_dir (str):
                The directory where the processed data is stored.
            movies_csv_file (str):
                The filename of processed movies csv file.
                Defaults to "movies_processed.csv".

        Returns:
            None
        """

        self.project_dir = project_dir
        self.processed_data_dir = processed_data_dir
        self.movies_csv_file = movies_csv_file

    @staticmethod
    def _train_model(movies_df: pd.DataFrame) -> TrainedModel:
        """Trains the recommendation model components.

        Args:
            movies_df (pd.DataFrame):
                The processed movie DataFrame.

        Returns:
            TrainedModel:
                A dictionary containing trained model components.
        """

        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
        tfidf_matrix: csr_matrix = tfidf_vectorizer.fit_transform(
            movies_df["movie_profile"]
        )

        cosine_sim: ndarray = cosine_similarity(tfidf_matrix, tfidf_matrix)

        movie_id_to_index = pd.Series(
            movies_df.index.values, index=movies_df["movieId"]
        )

        model: TrainedModel = {
            "movies_df": movies_df,
            "tfidf_vectorizer": tfidf_vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "cosine_sim": cosine_sim,
            "movie_id_to_index": movie_id_to_index,
        }

        return model

    def run(self) -> None:
        """Executes the complete model training and saving pipeline.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If the specified processed movies CSV file does not exist.
        """

        self._verify_dataset_files_exist()

        output_dir = os.path.join(self.project_dir, "app/models")
        output_filepath = os.path.join(output_dir, "recommendation_model.pkl")

        os.makedirs(output_dir, exist_ok=True)

        movies_df = self._load_processed_data()
        model = self._train_model(movies_df)

        with open(output_filepath, "wb") as f:
            pickle.dump(model, f)

        print(f"Recommendation model saved to '{output_filepath}'")

    def _get_file_path(self, file_name: str) -> str:
        """Constructs the full path for a given file name.

        Args:
            file_name (str):
                The name of the file.

        Returns:
            str:
                The full path to the file.
        """

        return os.path.join(self.processed_data_dir, file_name)

    def _verify_dataset_files_exist(self) -> None:
        """Checks if all required processed dataset files exist.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If at least one of the required MovieLens dataset files does not exist.
        """

        required_files = {
            "processed_movies.csv": self.movies_csv_file,
        }

        missing_files = [
            expected_name
            for expected_name, actual_name in required_files.items()
            if not os.path.exists(self._get_file_path(actual_name))
        ]

        if missing_files:
            raise FileNotFoundError(
                f"The following required files are missing: {', '.join(missing_files)}"
            )

    def _load_processed_data(self) -> pd.DataFrame:
        """Loads processed data from CSV file.

        Returns:
            pd.DataFrame:
                A DataFrame containing the processed movie data.
        """

        movies_path = self._get_file_path(self.movies_csv_file)

        movies_df = pd.read_csv(movies_path)

        return movies_df
