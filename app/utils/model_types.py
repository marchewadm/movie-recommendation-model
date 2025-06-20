from typing import TypedDict

from numpy import ndarray
from pandas import DataFrame, Series


class TrainedModel(TypedDict):
    """A TypedDict to define the structure of the trained recommendation model.

    Attributes:
        movies_df (DataFrame):
            The DataFrame containing movie metadata.
        cosine_sim (ndarray):
            The cosine similarity matrix.
        movie_id_to_index (Series):
            A mapping from movie ID to DataFrame index.
    """

    movies_df: DataFrame
    cosine_sim: ndarray
    movie_id_to_index: Series


class OutputData(TypedDict):
    """Represents the structure for data intended to be saved as an output file.

    Attributes:
        dataframe (DataFrame):
            The pandas DataFrame containing the data to be saved.
        filename (str):
            The name of the file (including extension) under which the DataFrame
            should be saved.
            For example, "movies_processed.csv".
    """

    dataframe: DataFrame
    filename: str
