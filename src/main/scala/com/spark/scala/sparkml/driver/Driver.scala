package com.spark.scala.sparkml.driver

import org.apache.spark.sql.SparkSession
import com.spark.scala.sparkml.linearregression.LinearRegression

object Driver extends App{
  
   println("Spark Session is being created")
  
   val spark=SparkSession.builder().master("local").appName("SparkML").getOrCreate()
   
   println("Spark Session has created")
   
   LinearRegression.executeLR(spark)
}