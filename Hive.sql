-- Databricks notebook source
SELECT
  time,
  sst(ACT_VAL, "-threshold 0.005") AS result
FROM
  key_controls
ORDER BY num ASC
;

-- COMMAND ----------


