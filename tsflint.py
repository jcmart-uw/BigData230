# Databricks notebook source
from pyspark.sql.functions import * 
#from pyspark.sql.types import *

import ts.flint
from ts.flint import FlintContext
flintContext = FlintContext(sqlContext)

df_control = flintContext.read.dataframe(spark.sql("select * from KEY_CONTROLS").where("RESULT_KEY_NBR = 11").select('DATE','ACTL_VAL') \
    .withColumn('time',unix_timestamp(col('DATE'), "yyyy-MM-dd").cast("timestamp")) \
    .select('time','ACTL_VAL').orderBy('time'))

df_control.show()

#df_control.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("dbfs:/FileStore/ActVal_Key11.csv")


# COMMAND ----------

from ts.flint import windows

df_control_previous_day_val = df_control.shiftTime(windows.future_absolute_time('1day')).toDF('time', 'previous_day_val')
df_control_previous_wk_val = df_control.shiftTime(windows.future_absolute_time('7day')).toDF('time', 'previous_wk_val')
df_control_joined = df_control.leftJoin(df_control_previous_day_val).leftJoin(df_control_previous_wk_val)
df_control_joined.show()

# COMMAND ----------

from ts.flint import summarizers

df_control_decayed_return = df_control_joined.where("time > '2018-06-15'").summarizeWindows(
    window = windows.past_absolute_time('42day'),
    summarizer = summarizers.ewma('previous_wk_val', alpha=0.5)
)

display(df_control_decayed_return)

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["previous_wk_val", "previous_wk_val_ewma"],
    outputCol="features")

output = assembler.transform(df_control_decayed_return).select('ACTL_VAL', 'features').toDF('label', 'features')

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

model = lr.fit(output)

# COMMAND ----------

model.summary.r2

# COMMAND ----------

display(model,output)

# COMMAND ----------


