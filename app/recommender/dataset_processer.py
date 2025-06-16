import os
import pandas as pd

from .base.runner import BaseRunner


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
            Defaults to "app/data/raw".
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
    """

    def __init__(
        self,
        project_dir: str,
        raw_data_dir: str = "app/data/raw",
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
                Defaults to "app/data/raw".
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

        Drops the "imdbId" column and converts "tmdbId" to nullable integer type.

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
            .rename(columns={"rating": "average_rating"})
        )

        rating_counts_df = (
            ratings_df.groupby("movieId")["rating"]
            .count()
            .reset_index()
            .rename(columns={"rating": "rating_count"})
        )

        return average_ratings_df, rating_counts_df

    @staticmethod
    def _merge_all(
        movies_df: pd.DataFrame,
        tags_df: pd.DataFrame,
        links_df: pd.DataFrame,
        average_ratings_df: pd.DataFrame,
        rating_counts_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merges all processed DataFrames into the main movies DataFrame.

        Performs left merges with tags, links, average ratings, and rating counts.
        Drops rows where "tmdbId" is missing, and fills NaN values in "tags",
        "average_rating", and "rating_count" columns.

        Args:
            movies_df (pd.DataFrame):
                The main movies DataFrame.
            tags_df (pd.DataFrame):
                DataFrame with processed tags.
            links_df (pd.DataFrame):
                DataFrame with processed links.
            average_ratings_df (pd.DataFrame):
                DataFrame with processed average ratings.
            rating_counts_df (pd.DataFrame):
                DataFrame with processed rating counts.

        Returns:
            pd.DataFrame:
                The merged movies DataFrame.
        """

        movies_df = movies_df.merge(tags_df, on="movieId", how="left")
        movies_df = movies_df.merge(links_df, on="movieId", how="left").dropna(
            subset=["tmdbId"]
        )
        movies_df = movies_df.merge(average_ratings_df, on="movieId", how="left")
        movies_df = movies_df.merge(rating_counts_df, on="movieId", how="left")

        movies_df["tags"] = movies_df["tags"].fillna("")
        movies_df["average_rating"] = movies_df["average_rating"].fillna(0.0)
        movies_df["rating_count"] = movies_df["rating_count"].fillna(0).astype(int)

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

    def run(self) -> None:
        """Executes the complete data processing pipeline.

        This method orchestrates the loading, cleaning, merging, and
        saving of the processed movie dataset. The final processed
        DataFrame is saved as "movies_processed.csv" in the
        "app/data/processed" directory within the project.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If at least one of the required MovieLens dataset files does not exist.
        """

        self._verify_dataset_files_exist()

        movies_df, ratings_df, links_df, tags_df = self._load_raw_data()

        movies_clean_df = self._process_data(movies_df, ratings_df, links_df, tags_df)
        movies_final_df = self._create_movie_profiles(movies_clean_df)

        self._save_output(movies_final_df, "movies_processed.csv")

    def _verify_dataset_files_exist(self) -> None:
        """Checks if all required MovieLens dataset files exist.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If at least one of the required MovieLens dataset files does not exist.
        """

        required_files = {
            "movies.csv": self.movies_csv_file,
            "ratings.csv": self.ratings_csv_file,
            "links.csv": self.links_csv_file,
            "tags.csv": self.tags_csv_file,
        }

        missing_files = [
            expected_name
            for expected_name, actual_name in required_files.items()
            if not os.path.exists(self._get_file_path(self.raw_data_dir, actual_name))
        ]

        if missing_files:
            raise FileNotFoundError(
                f"The following required files are missing: {', '.join(missing_files)}"
            )

    def _load_raw_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads raw data from CSV files.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                A tuple containing DataFrames for movies, ratings, links, and tags.
        """

        movies_path = self._get_file_path(self.raw_data_dir, self.movies_csv_file)
        ratings_path = self._get_file_path(self.raw_data_dir, self.ratings_csv_file)
        links_path = self._get_file_path(self.raw_data_dir, self.links_csv_file)
        tags_path = self._get_file_path(self.raw_data_dir, self.tags_csv_file)

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
    ) -> pd.DataFrame:
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
            pd.DataFrame:
                A single DataFrame with cleaned and merged movie data.
        """

        movies_df = self._clean_movies(movies_df)
        ratings_df = self._clean_ratings(ratings_df)
        tags_df = self._clean_tags(tags_df)
        links_df = self._clean_links(links_df)

        average_ratings_df, rating_counts_df = self._add_rating_stats(ratings_df)

        movies_df = self._merge_all(
            movies_df,
            tags_df,
            links_df,
            average_ratings_df,
            rating_counts_df,
        )

        return movies_df

    def _create_movie_profiles(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """Creates a "movie_profile" column by combining relevant text features.

        Generates a weighted "genres_weighted" column and then concatenates
        "title", "genres_weighted", and "tags" to form the "movie_profile".

        Args:
            movies_df (pd.DataFrame):
                A movies DataFrame containing "title", "genres", and "tags" columns.

        Returns:
            pd.DataFrame:
                DataFrame with the new "genres_weighted" and "movie_profile" columns.
        """

        movies_df["genres_weighted"] = (
            movies_df["genres"].apply(self._create_weighted_genres).str.strip()
        )

        movies_df["movie_profile"] = (
            movies_df["title"].fillna("")
            + " "
            + movies_df["genres_weighted"].fillna("")
            + " "
            + movies_df["tags"].fillna("")
        )

        return movies_df

    def _save_output(self, output_data: pd.DataFrame, file_name: str) -> None:
        """Saves the processed DataFrame to a CSV file.

        Args:
            output_data (pd.DataFrame):
                The processed DataFrame to be saved.
            file_name (str):
                The name of the file to save the DataFrame as.
                For example, "movies_processed.csv".

        Returns:
            None
        """

        output_dir = os.path.join(self.project_dir, "app/data/processed")
        output_filepath = os.path.join(output_dir, file_name)

        os.makedirs(output_dir, exist_ok=True)

        output_data.to_csv(output_filepath, index=False)

        print(f"Dataset processed and saved to '{output_filepath}'")
