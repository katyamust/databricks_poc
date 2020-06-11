import pandas as pd

from .data_loader import DataLoader

class KMeansSampleDataLoader(DataLoader):
    def download_dataset(self) -> None:
        pass

    def get_dataset(self):
       x = pd.read_csv("../kmeans_poc/data/x_poc.csv")
       return x