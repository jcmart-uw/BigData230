// Databricks notebook source
import org.apache.spark.sql.hive.HivemallOps._
import org.apache.spark.sql.hive.HivemallGroupedDataset._
import org.apache.spark.sql.hive.HivemallUtils._
import hivemall.xgboost.XGBoostOptions

// COMMAND ----------

import org.apache.spark.sql.hive.HiveContext
val hiveContext = new org.apache.spark.sql.hive.HiveContext(sc)
import hiveContext._

// COMMAND ----------

CREATE EXTERNAL TABLE timeseries (
  num INT,
  value DOUBLE
) ROW FORMAT DELIMITED
FIELDS TERMINATED BY '#'
STORED AS TEXTFILE
LOCATION '/FileStore/twitter/timeseries';

// COMMAND ----------


