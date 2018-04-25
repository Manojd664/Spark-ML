package com.spark.scala.sparkml.linearregression

import org.apache.spark.sql.SparkSession

object LinearRegression extends App{
  
  println("Spark Session is being created")
  
  val spark=SparkSession.builder().master("local").appName("SparkML").getOrCreate()
  
  println("Spark Session has created")
  
  println("Creating Dataframe for train Data")
  
  val trainDf=spark.read.format("csv").option("header","true").load("Inputfiles\\linearRegData\\train.csv")
  
  println("First 5 row of train data")
  
  trainDf.show(5)
  
}