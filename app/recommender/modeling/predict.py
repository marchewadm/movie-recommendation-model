import pickle

import pandas as pd
import numpy as np

from prettytable import PrettyTable

from app.utils.model_types import TrainedModel
from app.recommender.base.runner import BaseRunner


class MovieRecommenderEngine(BaseRunner):
    """Load a trained movie recommendation model and generate recommendations.

    Attributes:
        project_dir (str):
            The root directory of the project.
        model_dir (str):
            The directory where the trained recommendation model is stored.
        movie_id (int):
            The ID of the movie for which to find recommendations.
        trained_pkl_file (str):
            The filename of trained recommendation model.
            Defaults to "recommendation_model.pkl".
    """

    def __init__(
        self,
        project_dir: str,
        model_dir: str,
        movie_id: int,
        trained_pkl_file: str = "recommendation_model.pkl",
    ) -> None:
        """Initializes the MovieRecommenderEngine with file paths and directories.

        Args:
            project_dir (str):
                The root directory of the project.
            model_dir (str):
                The directory where the trained recommendation model is stored.
            movie_id (int):
                The ID of the movie for which to find recommendations.
            trained_pkl_file (str):
                The filename of trained recommendation model.
                Defaults to "recommendation_model.pkl".
        """

        super().__init__(project_dir)
        self.model_dir = model_dir
        self.movie_id = movie_id
        self.trained_pkl_file = trained_pkl_file

    @staticmethod
    def _read_data_from_model(
        model: TrainedModel,
    ) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
        """Extracts key data components from the loaded trained model.

        Args:
            model (TrainedModel):
                The loaded trained recommendation model as a TypedDict.

        Returns:
            tuple[pd.DataFrame, np.ndarray, pd.Series]:
                The extracted key data components from the loaded model:
                - pd.DataFrame: The DataFrame of movie metadata.
                - np.ndarray: The cosine similarity matrix.
                - pd.Series: The mapping from movie ID to DataFrame index.
        """

        movies_df = model["movies_df"]
        cosine_sim = model["cosine_sim"]
        movie_id_to_index = model["movie_id_to_index"]

        return movies_df, cosine_sim, movie_id_to_index

    @staticmethod
    def _find_movie_recommendations(
        movie_id: int,
        movie_index_map: pd.Series,
        movies_dataframe: pd.DataFrame,
        similarity_matrix: np.ndarray,
        n_recommendations: int = 10,
        target_genres: list[str] = None,
        target_tags: list[str] = None,
        min_rating_count: int = 50,
    ) -> pd.DataFrame:
        """Recommends movies similar to the given movie based on cosine similarity.

        The function uses a precomputed cosine similarity matrix to find movies most
        similar to the one specified by `movie_id`. Results can be filtered by genre,
        tag, and minimum number of ratings.

        Args:
            movie_id (int):
                The ID of the movie to find recommendations for.
            movie_index_map (pd.Series):
                A mapping from movieId to the corresponding row index in the similarity
                matrix and movies DataFrame.
            movies_dataframe (pd.DataFrame):
                The DataFrame containing movie metadata, such as title, genres, tags,
                average ratings, and rating count.
            similarity_matrix (np.ndarray):
                Matrix of cosine similarity scores between movies.
            n_recommendations (int):
                Number of recommendations to return.
                Defaults to 10.
            target_genres (list[str] | None):
                List of genres to filter recommendations by. If provided, only movies
                that share at least one of the specified genres will be returned.
            target_tags (list[str] | None):
                List of tags to filter recommendations by. If provided, only movies
                that share at least one of the specified tags will be returned.
            min_rating_count (int):
                Minimum number of ratings a recommended movie must have. This helps
                filter out obscure movies with unreliable averages.
                Defaults to 50.

        Returns:
            pd.DataFrame:
                A DataFrame containing recommended movies.
        """

        if movie_id not in movie_index_map:
            raise ValueError(f"Movie ID {movie_id} not found in the dataset.")

        recommendations = []

        target_genres_lower = (
            [genre.strip().lower() for genre in target_genres] if target_genres else []
        )
        target_tags_lower = (
            [tag.strip().lower() for tag in target_tags] if target_tags else []
        )

        movie_index = movie_index_map[movie_id]

        similarity_scores = list(enumerate(similarity_matrix[movie_index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[
            1:
        ]  # Remove the first row, as it always is the movie provided by the user.

        for similar_movie_index, similarity_score in similarity_scores:
            movie = movies_dataframe.iloc[similar_movie_index].copy()

            genres = (
                [genre.strip().lower() for genre in movie["genres"].split("|") if genre]
                if pd.notna(movie["genres"])
                else []
            )

            tags = (
                [tag.strip().lower() for tag in movie["tags"].split() if tag]
                if pd.notna(movie["tags"])
                else []
            )

            if target_genres_lower and not any(
                genre in genres for genre in target_genres_lower
            ):
                continue
            if target_tags_lower and not any(tag in tags for tag in target_tags_lower):
                continue
            if movie["ratingCount"] < min_rating_count:
                continue

            movie["similarityScore"] = similarity_score
            recommendations.append(movie)

            if len(recommendations) >= n_recommendations:
                break

        if not recommendations:
            raise LookupError(f"No recommendations found for movie ID {movie_id}.")

        recommendations_df = pd.DataFrame(recommendations).sort_values(
            by=["similarityScore", "averageRating"], ascending=[False, False]
        )

        return recommendations_df

    @staticmethod
    def _display_recommendations(recommendations: pd.DataFrame) -> None:
        """Displays the movie recommendations in a formatted PrettyTable.

        Args:
            recommendations (pd.DataFrame):
                A DataFrame containing recommended movies with columns:
                - "movieId",
                - "tmdbId",
                - "title",
                - "tags",
                - "genres",
                - "averageRating",
                - "ratingCount",
                - "similarityScore".

        Returns:
            None
        """

        table = PrettyTable()
        table.field_names = [
            "Movie ID",
            "TMDB ID",
            "Title",
            "Genres",
            "Average Rating",
            "Rating Count",
            "Similarity Score",
        ]

        columns_to_display = [
            "movieId",
            "tmdbId",
            "title",
            "genres",
            "averageRating",
            "ratingCount",
            "similarityScore",
        ]

        display_df: pd.DataFrame = recommendations[columns_to_display].copy()
        display_df["similarityScore"] = display_df["similarityScore"].apply(
            lambda x: f"{x:.2f}"
        )

        table.add_rows(display_df.to_numpy().tolist())

        print(table)

    def run(self) -> None:
        """Executes the movie recommendation process for a given movie ID.

        This method orchestrates loading the recommendation model, extracting its
        components, finding similar movie recommendations, and displaying them in a
        formatted table.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If at least one of the required models does not exist.
        """

        self._verify_required_files_exist(
            self.model_dir, {"recommendation_model.pkl": self.trained_pkl_file}
        )
        model = self._load_model()

        movies_df, cosine_sim, movie_id_to_index = self._read_data_from_model(model)

        recommendations = self._find_movie_recommendations(
            self.movie_id, movie_id_to_index, movies_df, cosine_sim
        )

        self._display_recommendations(recommendations)

    def _load_model(self) -> TrainedModel:
        """Loads the trained recommendation model from a pickle file.

        Returns:
            A TypedDict containing the loaded model components.
        """

        with open(
            self._get_file_path(self.model_dir, self.trained_pkl_file), "rb"
        ) as f:
            model: TrainedModel = pickle.load(f)

        return model
