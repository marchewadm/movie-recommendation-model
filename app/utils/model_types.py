from typing import TypedDict

from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix
from numpy import ndarray


# TODO: tfidf_vectorizer, tfidf_matrix could possibly be removed
class TrainedModel(TypedDict):
    """A TypedDict to define the structure of the trained recommendation model.

    Attributes:
        movies_df (DataFrame):
            The DataFrame containing movie metadata.
        tfidf_vectorizer (TfidfVectorizer):
            The fitted TF-IDF vectorizer.
        tfidf_matrix (csr_matrix):
            The TF-IDF feature matrix.
        cosine_sim (ndarray):
            The cosine similarity matrix.
        movie_id_to_index (Series):
            A mapping from movie ID to DataFrame index.
    """

    movies_df: DataFrame
    tfidf_vectorizer: TfidfVectorizer
    tfidf_matrix: csr_matrix
    cosine_sim: ndarray
    movie_id_to_index: Series
