import pytest
import pandas as pd

from kmeans_poc.models.k_means_sample_model import KMeansModel
from kmeans_poc.data.generate_dataset import generate_x

from pyspark.sql import SparkSession
from kmeans_poc.data.k_means_sample_data_loader import KMeansSampleDataLoader


@pytest.fixture
def mock_x():
    return generate_x()


def test_k_means_poc_model_can_fit(mock_x):

    model = KMeansModel()
    x = pd.DataFrame({"x":[1,2,3,4,5],"y":[11,12,13,14,15]})

    model.fit(x)

    assert model.k_means is not None


def test_k_means_poc_model_with_spark_dataframe(mock_x):
    model = KMeansModel()
    local_pyspark = SparkSession.builder.appName("test").getOrCreate()

    test_data_loader = KMeansSampleDataLoader(dataset_name="sample",
                                              dataset_version = 1.0,
                                              file  ="test_data/x_poc_for_test.csv",
                                              spark_session = local_pyspark)

    x = test_data_loader.get_dataset()
    model.fit(x)
    assert model.k_means is not None
