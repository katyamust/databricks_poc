# Databricks notebook source
# DBTITLE 1,Define where is your data 
dbutils.widgets.removeAll()
dbutils.widgets.text("storage_account_name", "", "A. Data lake gen2 storage account name")
dbutils.widgets.text("storage_account_key", "", "B. storage account key")
dbutils.widgets.text("container_name", "experiment1", "C. container")
dbutils.widgets.text("input_file_name", "x_poc_original.csv", "D. input file name")
dbutils.widgets.text("output_file_name", "x_poc.csv", "D. output file name")

account_name = dbutils.widgets.get("storage_account_name") 
account_key = dbutils.widgets.get("storage_account_key")
container = dbutils.widgets.get("container_name")
input_file_name = dbutils.widgets.get("input_file_name")
output_file_name = dbutils.widgets.get("output_file_name")

# COMMAND ----------

# DBTITLE 1,Define python package(kmeans_poc) imports
from kmeans_poc.data import KMeansSampleDataLoader
from kmeans_poc.models import KMeansModel
from kmeans_poc.data_processing import DataProcessor
from kmeans_poc.experimentation import MlflowExperimentation
from kmeans_poc.evaluation import KMeansSampleEvaluationMetrics, KMeansSampleEvaluator

# COMMAND ----------

# DBTITLE 1,Load your data
data_loader = KMeansSampleDataLoader(dataset_name="generated",dataset_version=1.0 )

#define function get_dataset
# x = data_loader.get_dataset() as part of package

file_location = "wasbs://"+ container + "@" + account_name + ".blob.core.windows.net/" + input_file_name
spark.conf.set( "fs.azure.account.key." + account_name + ".blob.core.windows.net", account_key)

# read the data
df = spark.read.format("csv").option("inferSchema", "true").load(file_location, header='true')

x = df.toPandas()
print(x.shape)

# COMMAND ----------

# DBTITLE 1,Define model
my_model = KMeansModel(n_clusters=1,viz_file="/dbfs/predicted_clusters.png")

# COMMAND ----------

# DBTITLE 1,Define evaluation criteria
evaluator = KMeansSampleEvaluator(my_model)

# COMMAND ----------

print(x.shape)

# COMMAND ----------

#my_model.fit(x)
#c = my_model.predict(x)
#evaluator.evaluate(x,c)

# COMMAND ----------

# DBTITLE 1,Define experimentation
experimentation = MlflowExperimentation(tracking_uri="databricks",files_to_log=[my_model.viz_file])

# COMMAND ----------

# DBTITLE 1,Fit model
my_model.fit(x)

# COMMAND ----------

# DBTITLE 1,Run experiment
from kmeans_poc import ExperimentRunner

experiment_runner = ExperimentRunner(
    model=my_model,
    X_train=x,
    y_train=None,
    X_test=x,
    y_test=x,
    data_loader=data_loader,
    log_experiment=True,
    experiment_logger=experimentation,
    evaluator=evaluator,
    experiment_name= "/my_experiment",
)

results = experiment_runner.run()
print(results)
