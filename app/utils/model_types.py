from typing import TypedDict

from pandas import DataFrame, Series
from numpy import ndarray


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
