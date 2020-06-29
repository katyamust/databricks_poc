from kmeans_poc.evaluation import Evaluator
from kmeans_poc.evaluation import KMeansSampleEvaluationMetrics
from kmeans_poc.visualization import plot_clusters


class KMeansSampleEvaluator(Evaluator):
    """
    Class to hold the logic for how the model is evaluated. For this Kmeans returns sse
    """

    def __init__(self, my_fitted_model):
        self.kmeans_fitted_model = my_fitted_model
        self.plot_file = my_fitted_model.viz_file
        super().__init__()

    def evaluate(self, X, _predicted_clusters) -> KMeansSampleEvaluationMetrics:
        sse = self.kmeans_fitted_model.k_means.inertia_
        plot_clusters(X, _predicted_clusters, self.plot_file)
        return KMeansSampleEvaluationMetrics(sse)