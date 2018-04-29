package com.spark.scala.sparkml.linearregression

import org.apache.spark.sql.SparkSession

object LinearRegression {

  def executeLR(spark: SparkSession) {

    println("Linear Regression execution has started")

    println("Creating Dataframe for train Data")

    val trainDf = spark.read.format("csv").option("header", "true").load("Inputfiles\\linearRegData\\train.csv")

    println("First 5 row of train data")

    trainDf.show(5)
    
    println("Creating Dataframe for test data")

    val testDf = spark.read.format("csv").option("header", "true").load("Inputfiles\\linearRegData\\test.csv")

    println("First 5 rows of test data")

    testDf.show(5)

  }

}