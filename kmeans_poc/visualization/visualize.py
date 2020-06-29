import matplotlib.pyplot as plt
# %matplotlib  inline

def plot_clusters (X, clusters, file_name):
    plt.scatter(X["x"], X["y"], c=clusters)
    plt.savefig(file_name)
    plt.show()
