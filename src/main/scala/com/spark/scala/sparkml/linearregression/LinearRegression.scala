package com.spark.scala.sparkml.linearregression

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.LinearRegression

object LinearRegression {

  def executeLR(spark: SparkSession) {

    println("Linear Regression execution has started")

    println("Creating Dataframe for train Data")

    var trainDf = spark.read.format("csv").option("header", "true").load("Inputfiles\\linearRegData\\train.csv")

    println("First 5 row of train data")

    trainDf.show(5)
    
    println("Creating Dataframe for test data")

    var testDf = spark.read.format("csv").option("header", "true").load("Inputfiles\\linearRegData\\test.csv")

    println("First 5 rows of test data")

    testDf.show(5)
    
    //Definig UDF to change category from string to integer
    val catgoryUDF=udf{(catgory:String)=>if(catgory=="A") 0 else if(catgory=="B") 1 else 2}
    //mapping Gender to integer values M->1 and F->0 and City_Category to integers A->0,B->1 and C->2
    trainDf=trainDf.withColumn("Gender", when(col("Gender")==="M",1).otherwise(0)).withColumn("City_Category",catgoryUDF(trainDf("City_Category")))
    
    testDf=testDf.withColumn("Gender", when(col("Gender")==="M",1).otherwise(0)).withColumn("City_Category",catgoryUDF(testDf("City_Category")))
    
    //Converting age column from string to integer
    val ageUDF=udf{(age:String)=>if(age.charAt(age.length()-1)=='+') age.slice(0,age.lastIndexOf("+")).toInt else {val ageStr=age.split("-"); ((ageStr(0).toInt+ageStr(1).toInt)/2).toInt}}
    
    trainDf=trainDf.withColumn("Age", ageUDF(trainDf("Age")))
    testDf=testDf.withColumn("Age", ageUDF(testDf("Age")))  
    
    
    //Removing P from product ID so the column Product Id will become integer and removing + from Stay_In_Current_City_Years column
    
    val removeFirstCharOrLastCharUDF=udf((str:String)=>if(str.charAt(0)=='P') str.slice(1,str.length).toInt else if(str.charAt(str.length()-1)=='+') str.slice(0,str.length()-1).toInt else str.toInt)
    
    trainDf=trainDf.withColumn("Product_ID",removeFirstCharOrLastCharUDF(trainDf("Product_ID"))).withColumn("Stay_In_Current_City_Years", removeFirstCharOrLastCharUDF(trainDf("Stay_In_Current_City_Years")))
    testDf=testDf.withColumn("Product_ID",removeFirstCharOrLastCharUDF(testDf("Product_ID"))).withColumn("Stay_In_Current_City_Years", removeFirstCharOrLastCharUDF(testDf("Stay_In_Current_City_Years")))
    
    //filling null values of Product_Category_2, Product_Category_3 with average value(assuming average may be 8)
    
    trainDf=trainDf.na.fill(Map("Product_Category_2"->8,"Product_Category_3"->8))
    
    testDf=testDf.na.fill(Map("Product_Category_2"->8,"Product_Category_3"->8))
    //List of trainDf columns
    //List(User_ID, Product_ID, Gender, Age, Occupation, City_Category, Stay_In_Current_City_Years, Marital_Status, Product_Category_1, Product_Category_2, Product_Category_3, Purchase)
    //Creating RFoumula of features and label columns
    val formula = new RFormula().setFormula("""Purchase ~ Product_ID+Gender+Occupation+City_Category+Stay_In_Current_City_Years+Marital_Status+Product_Category_1
                                            +Product_Category_2+Product_Category_3""").setFeaturesCol("features").setLabelCol("label")
       
    //Removing User_ID columns from trainDf and testDf
    trainDf=trainDf.drop("User_ID")
    var testIdRDD=testDf.select(testDf.col("User_ID")).rdd
    testDf=testDf.drop("User_ID")
    
    val strToInt=udf((str:String)=>if(str==null) 0 else str.toInt)
    trainDf=trainDf.withColumn("Occupation", strToInt(col("Occupation"))).withColumn("Marital_Status", strToInt(col("Marital_Status"))).withColumn("Product_Category_1", strToInt(col("Product_Category_1")))
    .withColumn("Product_Category_2", strToInt(col("Product_Category_2"))).withColumn("Product_Category_3", strToInt(col("Product_Category_3"))).
    withColumn("Purchase", strToInt(col("Purchase")))
    
     testDf=testDf.withColumn("Occupation", strToInt(col("Occupation"))).withColumn("Marital_Status", strToInt(col("Marital_Status"))).withColumn("Product_Category_1", strToInt(col("Product_Category_1")))
    .withColumn("Product_Category_2", strToInt(col("Product_Category_2"))).withColumn("Product_Category_3", strToInt(col("Product_Category_3")))
    
    val train=formula.fit(trainDf).transform(trainDf)  
    val test=formula.fit(testDf).transform(testDf)  
    
    
    //Instantiating LinearRegression 
    val linearRegression = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    val linearRegressionModel = linearRegression.fit(train)
    
    println("Predicting the Purchase..............")
    var pred=linearRegressionModel.transform(test)
    
    val prediction=pred.select("prediction").rdd 
    
    //Assigning prediction to the user_id
    val finalAns=testIdRDD.zip(prediction).map(x=>(x._1(0),x._2(0)))
    
    finalAns.collect.foreach(println)
    
    //Sample Output: 
    /*
 (User_ID,Purchase)
(1001738,9054.908765231517)
(1001741,10141.681889345686)
(1001741,10885.967114730061)
(1001741,8125.42378656851)
(1001743,11025.010401425843)
(1001743,4406.709756356637)
(1001744,9361.751317715636)
(1001746,10217.029755117721)
(1001748,10029.396999679395)
(1001748,8499.522684501968)
(1001749,9958.799356692085)
(1001752,8938.824926638317)
(1001752,7891.099866897916)
(1001752,9957.08781066167)
(1001753,8148.507125198994)
(1001755,7858.888196518104)
(1001755,10745.509404915332)
(1001755,8283.961576127534)
(1001757,10575.443819398512)
(1001758,8829.185975089178)
(1001758,9235.069803581056)
(1001758,7172.470580628603)
(1001759,11217.267256965544)
(1001759,13040.184559045932)
(1001759,9344.547912830621)
(1001759,10158.949041532087)
(1001764,6441.824348434387)
(1001767,10674.70173223989)*/
    
  }

}