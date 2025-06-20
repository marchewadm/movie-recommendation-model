import os
import pickle

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from numpy import ndarray

from app.utils.model_types import TrainedModel
from app.recommender.base.runner import BaseRunner


class MovieRecommenderTrainer(BaseRunner):
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

        super().__init__(project_dir)
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
            movies_df["movieProfile"]
        )

        cosine_sim: ndarray = cosine_similarity(tfidf_matrix, tfidf_matrix)

        movie_id_to_index = pd.Series(
            movies_df.index.values, index=movies_df["movieId"]
        )

        model: TrainedModel = {
            "movies_df": movies_df,
            "cosine_sim": cosine_sim,
            "movie_id_to_index": movie_id_to_index,
        }

        return model

    @staticmethod
    def _filter_columns_for_export(movies_df: pd.DataFrame) -> pd.DataFrame:
        """Filters out unnecessary columns before exporting the DataFrame to CSV.

        Args:
            movies_df (pd.DataFrame):
                The input DataFrame containing movie data, potentially including
                "movieProfile" column.

        Returns:
            pd.DataFrame:
                A new DataFrame with the "movieProfile" column removed.
        """

        return movies_df.drop("movieProfile", axis=1)

    def run(self) -> None:
        """Executes the complete model training and saving pipeline.

        This method orchestrates loading of the processed movie data, training and
        saving the model. The model is saved as "recommendation_model.pkl" in
        "app/models/" directory within the project.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If the specified processed movies CSV file does not exist.
        """

        self._verify_required_files_exist(
            self.processed_data_dir,
            {
                "movies_processed.csv": self.movies_csv_file,
            },
        )

        movies_df = self._load_processed_data()
        model = self._train_model(movies_df)

        model["movies_df"] = self._filter_columns_for_export(model["movies_df"])

        self._save_output(model)

    def _load_processed_data(self) -> pd.DataFrame:
        """Loads processed data from CSV file.

        Returns:
            pd.DataFrame:
                A DataFrame containing the processed movie data.
        """

        movies_path = self._get_file_path(self.processed_data_dir, self.movies_csv_file)

        movies_df = pd.read_csv(movies_path)

        return movies_df

    def _save_output(
        self, output_data: TrainedModel, file_name: str = "recommendation_model.pkl"
    ) -> None:
        """Saves the trained recommendation model to a pickle file.

        Args:
            output_data (TrainedModel):
                The trained model object to be saved.
            file_name (str):
                The name of the file to save the model as.
                Defaults to "recommendation_model.pkl".

        Returns:
            None
        """

        output_dir = os.path.join(self.project_dir, "app/models")
        output_filepath = os.path.join(output_dir, file_name)

        os.makedirs(output_dir, exist_ok=True)

        with open(output_filepath, "wb") as f:
            pickle.dump(output_data, f)

        print(f"Recommendation model saved to '{output_filepath}'")
