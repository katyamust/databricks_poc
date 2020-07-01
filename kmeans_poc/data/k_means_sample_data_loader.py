import pandas as pd

from .data_loader import DataLoader


class KMeansSampleDataLoader(DataLoader):

    def __init__(self, dataset_name, dataset_version, file=None, spark_session=None):
        super().__init__(dataset_name,dataset_version,name=None, file=file, spark_session=spark_session)

    def download_dataset(self) -> None:
        pass

    def get_dataset(self) -> pd.DataFrame :
        file_path = self.data_params["file"]
        spark = self.data_params["spark_session"]
        df = spark.read.format("csv").load(file_path, header='true')
        return df.toPandas()