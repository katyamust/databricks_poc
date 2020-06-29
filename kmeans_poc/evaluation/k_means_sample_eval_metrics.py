from kmeans_poc.evaluation import EvaluationMetrics


class KMeansSampleEvaluationMetrics(EvaluationMetrics):
    """
    Class to hold the actual values the evaluation created, e.g. precision, recall, MSE.
    in this sample: evaluation metric is sse: sum of squared distance between points and
    centroids within cluster
    """

    def __init__(self, sse):
        self.sse = sse
        super().__init__()

    def get_metrics(self):
        return {"sse": self.sse}

    def __repr__(self):
        return f"sse: {self.sse}"