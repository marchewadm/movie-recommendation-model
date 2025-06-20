import os

import pandas as pd

from app.utils.model_types import OutputData
from app.recommender.base.runner import BaseRunner


class DatasetProcesser(BaseRunner):
    """Processes raw MovieLens dataset files.

    This class handles loading, cleaning, and merging movie-related data from
    various CSV files (movies, ratings, links, tags) into a single CSV file
    for further analysis.

    Attributes:
        project_dir (str):
            The root directory of the project.
        raw_data_dir (str):
            The directory where the raw MovieLens dataset CSV files are located.
        movies_csv_file (str):
            The filename of the movies CSV file.
        ratings_csv_file (str):
            The filename of the ratings CSV file.
        links_csv_file (str):
            The filename of the links CSV file.
        tags_csv_file (str):
            The filename of the tags CSV file.
    """

    def __init__(
        self,
        project_dir: str,
        raw_data_dir: str,
        movies_csv_file: str = "movies.csv",
        ratings_csv_file: str = "ratings.csv",
        links_csv_file: str = "links.csv",
        tags_csv_file: str = "tags.csv",
    ) -> None:
        """Initializes the DatasetProcesser with file paths and directories.

        Args:
            project_dir (str):
                The root directory of the project.
            raw_data_dir (str):
                The directory where the raw MovieLens dataset CSV files are located.
            movies_csv_file (str):
                The filename of the movies CSV file.
                Defaults to "movies.csv".
            ratings_csv_file (str):
                The filename of the ratings CSV file.
                Defaults to "ratings.csv".
            links_csv_file (str):
                The filename of the links CSV file.
                Defaults to "links.csv".
            tags_csv_file (str):
                The filename of the tags CSV file.
                Defaults to "tags.csv".

        Returns:
            None
        """

        super().__init__(project_dir)

        self.raw_data_dir = raw_data_dir
        self.movies_csv_file = movies_csv_file
        self.ratings_csv_file = ratings_csv_file
        self.links_csv_file = links_csv_file
        self.tags_csv_file = tags_csv_file

    @staticmethod
    def _drop_parenthesis(string: str) -> str:
        """Removes all parentheses from the input string.

        Args:
            string (str):
                A string potentially containing "(" and/or ")".

        Returns:
            str:
                A copy of the input string with all "(" and ")" characters removed.
        """

        return string.replace("(", "").replace(")", "")

    @staticmethod
    def _clean_ratings(ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the ratings DataFrame.

        Drops the "timestamp" column.

        Args:
            ratings_df (pd.DataFrame):
                The input ratings DataFrame.

        Returns:
            pd.DataFrame:
                The cleaned ratings DataFrame.
        """

        ratings_df = ratings_df.drop("timestamp", axis=1)

        return ratings_df

    @staticmethod
    def _clean_links(links_df: pd.DataFrame) -> pd.DataFrame:
        """Cleansing the links DataFrame.

        Drops the "imdbId" column and converts "tmdbId" to a nullable integer type.

        Args:
            links_df (pd.DataFrame):
                The input links DataFrame.

        Returns:
            pd.DataFrame:
                The cleaned links DataFrame.
        """

        links_df = links_df.drop("imdbId", axis=1)

        links_df["tmdbId"] = links_df["tmdbId"].astype("Int64")

        return links_df

    @staticmethod
    def _merge_initial_data(
        movies_df: pd.DataFrame,
        tags_df: pd.DataFrame,
        links_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merges movies, tags, and links DataFrames.

        Args:
            movies_df (pd.DataFrame):
                The main movies DataFrame.
            tags_df (pd.DataFrame):
                DataFrame with processed tags.
            links_df (pd.DataFrame):
                DataFrame with processed links.

        Returns:
            pd.DataFrame:
                The merged movies DataFrame.
        """

        movies_df = movies_df.merge(tags_df, on="movieId", how="left")
        movies_df = movies_df.merge(links_df, on="movieId", how="left").dropna(
            subset=["tmdbId"]
        )

        movies_df["tags"] = movies_df["tags"].fillna("")

        return movies_df

    def run(self) -> None:
        """Executes the complete data processing pipeline.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If at least one of the required MovieLens dataset files does not exist.
        """

        self._verify_required_files_exist(
            self.raw_data_dir,
            {
                "movies.csv": self.movies_csv_file,
                "ratings.csv": self.ratings_csv_file,
                "links.csv": self.links_csv_file,
                "tags.csv": self.tags_csv_file,
            },
        )

        movies_df, ratings_df, links_df, tags_df = self._load_raw_data()

        movies_df, ratings_df = self._process_data(
            movies_df, ratings_df, links_df, tags_df
        )

        movies_output: OutputData = {
            "dataframe": movies_df,
            "filename": "movies_interim.csv",
        }
        ratings_output: OutputData = {
            "dataframe": ratings_df,
            "filename": "ratings_interim.csv",
        }

        self._save_output(movies_output, ratings_output)

    def _load_raw_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads raw data from CSV files.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                A tuple containing DataFrames for movies, ratings, links, and tags.
        """

        movies_path = self._join_paths(self.raw_data_dir, self.movies_csv_file)
        ratings_path = self._join_paths(self.raw_data_dir, self.ratings_csv_file)
        links_path = self._join_paths(self.raw_data_dir, self.links_csv_file)
        tags_path = self._join_paths(self.raw_data_dir, self.tags_csv_file)

        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
        links_df = pd.read_csv(links_path)
        tags_df = pd.read_csv(tags_path)

        return movies_df, ratings_df, links_df, tags_df

    def _unique_words_preserve_order(self, tags_series: pd.Series) -> str:
        """Concatenates unique words from a Series of tags while preserving their order.

        This function processes a pandas Series of tag strings, removes any
        parentheses, splits each tag into individual words, and returns a single
        string of unique words in the order they appear.

        Args:
            tags_series (pd.Series):
                A Series containing tag strings associated with a movie

        Returns:
            str:
                A single space-separated string of unique words in original order.
        """

        words = []
        seen = set()

        for tag in tags_series:
            tag = self._drop_parenthesis(tag)

            for word in tag.split():
                if word not in seen:
                    seen.add(word)
                    words.append(word)

        return " ".join(words)

    def _clean_movies(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the movies DataFrame.

        Removes parentheses from the "title" column.

        Args:
            movies_df (pd.DataFrame):
                The input movies DataFrame.

        Returns:
            pd.DataFrame:
                The cleaned movies DataFrame.
        """

        movies_df["title"] = movies_df["title"].apply(self._drop_parenthesis)

        return movies_df

    def _clean_tags(self, tags_df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the tags DataFrame.

        Drops "timestamp" and "userId" columns, fills NaNs in "tag",
        converts "tag" to lowercase and strips whitespace, and then
        groups tags by movieId, concatenating unique words while preserving order.

        Args:
            tags_df (pd.DataFrame):
                The input tags DataFrame.

        Returns:
            pd.DataFrame:
                The cleaned tags DataFrame.
        """

        tags_df = tags_df.drop(["timestamp", "userId"], axis=1)

        tags_df["tag"] = tags_df["tag"].fillna("").astype(str).str.lower().str.strip()

        tags_df = (
            tags_df.groupby("movieId")["tag"]
            .apply(self._unique_words_preserve_order)
            .reset_index()
            .rename(columns={"tag": "tags"})
        )

        return tags_df

    def _process_data(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        links_df: pd.DataFrame,
        tags_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Cleans and merges the input DataFrames into a single processed DataFrame.

        Args:
            movies_df (pd.DataFrame):
                The input movies DataFrame.
            ratings_df (pd.DataFrame):
                The input ratings DataFrame.
            links_df (pd.DataFrame):
                The input links DataFrame.
            tags_df (pd.DataFrame):
                The input tags DataFrame.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                A tuple containing movies and ratings DataFrames.
        """

        movies_df = self._clean_movies(movies_df)
        ratings_df = self._clean_ratings(ratings_df)
        tags_df = self._clean_tags(tags_df)
        links_df = self._clean_links(links_df)

        movies_df = self._merge_initial_data(
            movies_df,
            tags_df,
            links_df,
        )

        return movies_df, ratings_df

    def _save_output(
        self, movies_output: OutputData, ratings_output: OutputData
    ) -> None:
        """Saves the processed DataFrames to a CSV files.

        Args:
            movies_output (OutputData):
                The processed movies DataFrame to be saved.
            ratings_output (OutputData):
                The processed ratings DataFrame to be saved.

        Returns:
            None
        """

        output_dir = self._join_paths(self.project_dir, "app/data/interim")

        movies_filepath = self._join_paths(output_dir, movies_output["filename"])
        ratings_filepath = self._join_paths(output_dir, ratings_output["filename"])

        os.makedirs(output_dir, exist_ok=True)

        movies_output["dataframe"].to_csv(movies_filepath, index=False)
        ratings_output["dataframe"].to_csv(ratings_filepath, index=False)

        print(f"Movies dataset processed and saved to '{movies_filepath}'")
        print(f"Ratings dataset processed and saved to '{ratings_filepath}'")
