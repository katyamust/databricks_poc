from pyspark.sql import SparkSession
from kmeans_poc.data.k_means_sample_data_loader import KMeansSampleDataLoader


def test_spark_local_data_loader():

    local_pyspark = SparkSession.builder.appName("test").getOrCreate()

    test_data_loader = KMeansSampleDataLoader(dataset_name="sample",
                                              dataset_version = 1.0,
                                              file  ="test_data/x_poc_for_test.csv",
                                              spark_session = local_pyspark)

    df = test_data_loader.get_dataset()
    assert df.shape[0] == 100