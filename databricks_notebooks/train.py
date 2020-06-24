# Databricks notebook source
# DBTITLE 1,Create widget to enter account details
dbutils.widgets.removeAll()
dbutils.widgets.text("storage_account_name", "", "A. Data lake gen2 storage account name")
dbutils.widgets.text("storage_account_key", "", "B. storage account key")
dbutils.widgets.text("container_name", "root", "C. container")
dbutils.widgets.text("file_name", "x_poc.csv", "D. file name")

# COMMAND ----------

account_name = dbutils.widgets.get("storage_account_name") 
account_key = dbutils.widgets.get("storage_account_key")
container_name = dbutils.widgets.get("container_name")
file_name = dbutils.widgets.get("file_name")

# COMMAND ----------

file_location = "wasbs://"+ container + "@" + account_name + ".blob.core.windows.net/" + file_name
spark.conf.set( "fs.azure.account.key." + account_name + ".blob.core.windows.net", account_key)

# read the data
df = spark.read.format("csv").option("inferSchema", "true").load(file_location, header='true')
df.show()
pdf = df.toPandas()
print(pdf.head(5))


# COMMAND ----------

# DBTITLE 1,Fitting model and clustering
from kmeans_poc.models import KMeansModel
from kmeans_poc.data_processing import DataProcessor

best_model = KMeansModel(n_clusters=2)

best_model.fit(pdf)
predicted_clusters = best_model.predict(pdf)
print("Success")

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import *
from pyspark.sql.window import *

df = df.withColumn("id",  row_number().over(Window.orderBy(lit(1))) ) 
#df.show(100)

pdf_temp = pd.DataFrame(predicted_clusters, columns=["cluster"])
df_clusters = sqlContext.createDataFrame(pdf_temp).withColumn("id", row_number().over(Window.orderBy(lit(1))) )
#df_clusters.show(100)

df_enriched = df_clusters.join(df, "id", "outer").drop("id")
#df_enriched.show(100)

blobUrl =  "wasbs://root@datalakepocetisalat.blob.core.windows.net"

df_enriched.write \
.mode("overwrite") \
.format("com.databricks.spark.csv") \
.option("header", "true") \
.csv(blobUrl + "/reports/")

files = dbutils.fs.ls(blobUrl + '/reports/')
output_file = [x for x in files if x.name.startswith("part-")]
dbutils.fs.mv(output_file[0].path, "%s/testoutput11.csv" % (blobUrl))

#file_location = "wasbs://root@datalakepocetisalat.blob.core.windows.net/x_poc1.csv"

#df_enriched.write.mode("overwrite").save(file_location, format='csv')

#dfGPS.write.mode("overwrite").format("com.databricks.spark.csv").option("header","true").csv("/mnt/<mount-name>")