import pytest

from kmeans_poc.models.k_means_sample_model import KMeansModel
from kmeans_poc.data.generate_dataset import generate_x

@pytest.fixture
def mock_x():
    return generate_x()


def test_k_means_poc_model_can_fit(mock_x):

    model = KMeansModel()
    model.fit(mock_x)

    assert model.k_means is not None