import os

import pandas as pd

from app.utils.model_types import OutputData
from app.recommender.base.runner import BaseRunner


class FeatureEngineer(BaseRunner):
    """Performs feature engineering on interim movie and ratings data.

    Attributes:
        project_dir (str):
            The root directory of the project.
        interim_data_dir (str):
            The directory where the interim data files are located.
        movies_csv_file (str):
            The filename of the interim movies CSV file.
        ratings_csv_file (str):
            The filename of the interim ratings CSV file.
    """

    def __init__(
        self,
        project_dir: str,
        interim_data_dir: str,
        movies_csv_file: str = "movies_interim.csv",
        ratings_csv_file: str = "ratings_interim.csv",
    ) -> None:
        """Initializes the FeatureEngineer with file paths and directories.

        Args:
            project_dir (str):
                The root directory of the project.
            interim_data_dir (str):
                The directory where the interim data files are stored.
            movies_csv_file (str):
                The filename of the interim movies CSV file.
                Defaults to "movies_interim.csv".
            ratings_csv_file (str):
                The filename of the interim ratings CSV file.
                Defaults to "ratings_interim.csv".

        Returns:
            None
        """

        super().__init__(project_dir)

        self.interim_data_dir = interim_data_dir
        self.movies_csv_file = movies_csv_file
        self.ratings_csv_file = ratings_csv_file

    @staticmethod
    def _add_rating_stats(
        ratings_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calculates average ratings and rating counts for movies.

        Args:
            ratings_df (pd.DataFrame):
                The input ratings DataFrame.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                A tuple containing average ratings and the rating counts.
        """

        average_ratings_df = (
            ratings_df.groupby("movieId")["rating"]
            .mean()
            .reset_index()
            .rename(columns={"rating": "averageRating"})
        )

        rating_counts_df = (
            ratings_df.groupby("movieId")["rating"]
            .count()
            .reset_index()
            .rename(columns={"rating": "ratingCount"})
        )

        return average_ratings_df, rating_counts_df

    @staticmethod
    def _merge_features(
        movies_df: pd.DataFrame,
        average_ratings_df: pd.DataFrame,
        rating_counts_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merges rating statistics (average ratings and counts) into
        the movies DataFrame.

        Args:
            movies_df (pd.DataFrame):
                The main movies DataFrame.
            average_ratings_df (pd.DataFrame):
                DataFrame containing "movieId" and "averageRating".
            rating_counts_df (pd.DataFrame):
                DataFrame containing "movieId" and "ratingCount".

        Returns:
            pd.DataFrame:
                The movies DataFrame with
                "averageRating" and "ratingCount" columns added.
        """

        movies_df = movies_df.merge(average_ratings_df, on="movieId", how="left")
        movies_df = movies_df.merge(rating_counts_df, on="movieId", how="left")

        movies_df["averageRating"] = movies_df["averageRating"].fillna(0.0)
        movies_df["ratingCount"] = movies_df["ratingCount"].fillna(0).astype(int)

        return movies_df

    @staticmethod
    def _create_weighted_genres(genres: str | None, weight: int = 3) -> str:
        """Generates a space-separated string of genres repeated "weight" times.

        Cleans the input by removing hyphens and handles empty or missing input safely.

        Args:
            genres (str | None):
                A string containing genres separated by "|".
                For example, "Action|Sci-Fi|Drama".
            weight (int):
                How many times to repeat the genres.
                Defaults to 3.

        Returns:
            str:
                A space-separated string of weighted genres.
                For example, "Action SciFi Action SciFi Action SciFi".
        """

        if not isinstance(genres, str) or not genres.strip():
            return ""

        processed_genres = [
            genre.replace("-", "") for genre in genres.split("|") if genre.strip()
        ]

        return (" ".join(processed_genres) + " ") * weight

    @staticmethod
    def _filter_columns_for_export(movies_df: pd.DataFrame) -> pd.DataFrame:
        """Filters out unnecessary columns before exporting the DataFrame to CSV.

        Args:
            movies_df (pd.DataFrame):
                The input DataFrame containing movie data, potentially including
                "genresWeighted" column.

        Returns:
            pd.DataFrame:
                A new DataFrame with the "genresWeighted" column removed.
        """

        return movies_df.drop("genresWeighted", axis=1)

    def run(self) -> None:
        """Executes the complete feature engineering pipeline.

        This method orchestrates the loading of interim data, calculating
        rating statistics, merging various features into the movies DataFrame,
        creating movie profiles, and saving the final processed DataFrame
        to the "app/data/processed" directory.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If any of the required interim CSV files
                (movies or ratings) do not exist.

        """

        self._verify_required_files_exist(
            self.interim_data_dir,
            {
                "movies_interim.csv": self.movies_csv_file,
                "ratings_interim.csv": self.ratings_csv_file,
            },
        )

        movies_df, ratings_df = self._load_interim_data()

        average_ratings_df, rating_counts_df = self._add_rating_stats(ratings_df)

        movies_df = self._merge_features(
            movies_df, average_ratings_df, rating_counts_df
        )
        movies_df = self._create_movie_profiles(movies_df)
        movies_df = self._filter_columns_for_export(movies_df)

        movies_output: OutputData = {
            "dataframe": movies_df,
            "filename": "movies_processed.csv",
        }

        self._save_output(movies_output)

    def _load_interim_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads intermediate movie and ratings data from CSV files.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                A tuple containing movies and ratings DataFrames.
        """

        movies_path = self._join_paths(self.interim_data_dir, self.movies_csv_file)
        ratings_path = self._join_paths(self.interim_data_dir, self.ratings_csv_file)

        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)

        return movies_df, ratings_df

    def _create_movie_profiles(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Creates a "movieProfile" column by combining relevant text features.

        Generates a weighted "genresWeighted" column and then concatenates
        "title", "genresWeighted", and "tags" to form the "movieProfile".

        Args:
            movies_df (pd.DataFrame):
                A movies DataFrame containing "title", "genres", and "tags" columns.

        Returns:
            pd.DataFrame:
                DataFrame with the new "genresWeighted" and "movieProfile" columns.
        """

        movies_df["genresWeighted"] = (
            movies_df["genres"].apply(self._create_weighted_genres).str.strip()
        )

        movies_df["movieProfile"] = (
            movies_df["title"].fillna("")
            + " "
            + movies_df["genresWeighted"].fillna("")
            + " "
            + movies_df["tags"].fillna("")
        )

        return movies_df

    def _save_output(self, movies_output: OutputData) -> None:
        """Saves the processed movies DataFrame to a CSV file.

        Args:
            movies_output (OutputData):
                A TypedDict containing the DataFrame
                to be saved and its desired filename.

        Returns:
            None
        """

        output_dir = self._join_paths(self.project_dir, "app/data/processed")

        movies_filepath = self._join_paths(output_dir, movies_output["filename"])

        os.makedirs(output_dir, exist_ok=True)

        movies_output["dataframe"].to_csv(movies_filepath, index=False)

        print(f"Movies dataset processed and saved to '{movies_filepath}'")
