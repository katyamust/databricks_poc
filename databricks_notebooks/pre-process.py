# Databricks notebook source
dbutils.widgets.removeAll()
dbutils.widgets.text("storage_account_name", "", "A. Data lake gen2 storage account name")
dbutils.widgets.text("storage_account_key", "", "B. storage account key")
dbutils.widgets.text("container_name", "root", "C. container")
dbutils.widgets.text("input_file_name", "x_poc_original.csv", "D. input file name")
dbutils.widgets.text("output_file_name", "x_poc.csv", "D. output file name")

# COMMAND ----------

account_name = dbutils.widgets.get("storage_account_name") 
account_key = dbutils.widgets.get("storage_account_key")
container = dbutils.widgets.get("container_name")
input_file_name = dbutils.widgets.get("input_file_name")
output_file_name = dbutils.widgets.get("output_file_name")

# COMMAND ----------

file_location = "wasbs://"+ container + "@" + account_name + ".blob.core.windows.net/" + input_file_name
spark.conf.set( "fs.azure.account.key." + account_name + ".blob.core.windows.net", account_key)

# read the data
df = spark.read.format("csv").option("inferSchema", "true").load(file_location, header='true')
df.show()
pdf = df.toPandas()
print(pdf.head(5))
pdf = pdf.drop(df.columns[2], axis=1)

df_processed = sqlContext.createDataFrame(pdf)

df_processed.show()

# COMMAND ----------

blobUrl =  "wasbs://" + container + "@" + account_name + ".blob.core.windows.net"

df_processed.write \
.mode("overwrite") \
.format("com.databricks.spark.csv") \
.option("header", "true") \
.csv(blobUrl + "/reports/")

files = dbutils.fs.ls(blobUrl + '/reports/')
output_file = [x for x in files if x.name.startswith("part-")]
output_file_url = blobUrl + "/" + output_file_name

dbutils.fs.mv(output_file[0].path, output_file_url)