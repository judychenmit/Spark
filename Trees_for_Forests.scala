// Databricks notebook source exported at Fri, 19 Feb 2016 01:59:41 UTC
// MAGIC %md
// MAGIC # Assignment Problem 2: Analyze Forest Coverage 

// COMMAND ----------

// MAGIC %md 
// MAGIC #### Instructions 
// MAGIC **This is an Open-ended problem. I'm not looking for one correct answer, but an organized analysis report on the data. **
// MAGIC 
// MAGIC This is a very clean dataset great for classification. The data file contains 581,012 lines, each containing 55 fields. The first 54 fields are properties of a certain place on earth, the 55th field is the type of land coverage. Details of the fields in the README file below. 
// MAGIC 
// MAGIC 1. Use Spark to parse the file, prepare data for classification;
// MAGIC 1. Show some basic statistics of the data fields
// MAGIC 1. Build a Random Forest model in Spark to analyze the data. 
// MAGIC 1. Split the dataset to 70% and 30% for training and test dataset.  
// MAGIC 1. Train differnt classificiers and observe the performance/error rate of them. 
// MAGIC 1. Use Spark to do your calculations, then use dataframes to draw some plots. Describe each plot briefly and draw some conclusions.  

// COMMAND ----------

// MAGIC %md
// MAGIC #### How to work on and hand in this assignment
// MAGIC Simply clone this Notebook to your own directory. Write your analysis report, and send me the link (see address bar) of your Notebook before the assignment is due. 
// MAGIC 
// MAGIC If you prefer to do it in Python (or R), you can create a Python/R Notebook and send me the link to it. 

// COMMAND ----------

// MAGIC %md
// MAGIC #### Load the data into an RDD

// COMMAND ----------

//Set S3 

sc.hadoopConfiguration.set("fs.s3n.awsAccessKeyId", "AKIAJH57T"+"SADMXPN"+"3NWA")
sc.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey", "cl7ON3wPVCf"+"a42eAzHjRD"+"v0iVJgsApuS"+"H3qwyMwF")

//read txt file, gzipped files will be auto-unzipped
val myDataRDD = sc.textFile("s3n://mlonspark/covtype.data.gz")
val myReadmeRDD = sc.textFile("s3n://mlonspark/covtype.info")

println(myDataRDD.count())
myDataRDD.take(5).foreach(println)

// COMMAND ----------

// MAGIC %python
// MAGIC #Set S3 
// MAGIC hadoopConfig = sc._jsc.hadoopConfiguration()
// MAGIC hadoopConfig.set("fs.s3n.awsAccessKeyId", "AKIAJH57T"+"SADMXPN"+"3NWA")
// MAGIC hadoopConfig.set("fs.s3n.awsSecretAccessKey", "cl7ON3wPVCf"+"a42eAzHjRD"+"v0iVJgsApuS"+"H3qwyMwF")
// MAGIC 
// MAGIC #read txt file, gzipped files will be auto-unzipped
// MAGIC myDataRDD = sc.textFile("s3n://mlonspark/covtype.data.gz")
// MAGIC myReadmeRDD = sc.textFile("s3n://mlonspark/covtype.info")
// MAGIC 
// MAGIC print(myDataRDD.count())
// MAGIC print("\n".join(myDataRDD.take(5)))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Readme File

// COMMAND ----------

// MAGIC %python
// MAGIC #print('\n'.join(myReadmeRDD.collect()))
// MAGIC for x in myReadmeRDD.collect():
// MAGIC   print x

// COMMAND ----------

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils



// COMMAND ----------

// MAGIC %python
// MAGIC # Show some info on the data.
// MAGIC 
// MAGIC myDataRDD.count()

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.mllib.stat import Statistics
// MAGIC 
// MAGIC myDataRDD_quant = myDataRDD.map(lambda line: [line.split(',')[:9]])
// MAGIC myDataRDD_quant.take(5)
// MAGIC summary = Statistics.colStats(myDataRDD_quant)
// MAGIC #print('Elevation (m)', 'Aspect(degrees)','Slope(degrees)','Horizontal_Distance_To_Hydrology(m)','Vertical_Distance_To_Hydrology (m)', \ 
// MAGIC #      'Horizontal_Distance_To_Roadways (m)','Hillshade_9am (0 to 255 index)','Hillshade_Noon','Hillshade_3pm', \
// MAGIC #      'Horizontal_Distance_To_Fire_Points (m)')
// MAGIC print(summary.mean())
// MAGIC print(summary.variance())
// MAGIC print(summary.numNonzeros())

// COMMAND ----------

// MAGIC %python
// MAGIC WildernessAreas = myDataRDD.map(lambda x: x.split(',')[10])
// MAGIC WildernessAreas.take(10)
// MAGIC 
// MAGIC from operator import add
// MAGIC num_WildernessAreas = WildernessAreas.reduce(add)
// MAGIC num_WildernessAreas.take(1)

// COMMAND ----------

// MAGIC %python
// MAGIC # Split data aproximately into training (60%) and test (40%)
// MAGIC t1, t2 = myDataRDD.randomSplit([0.7, 0.3], seed=None)

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.mllib.regression import LabeledPoint
// MAGIC 
// MAGIC split_data_1 = t1.map(lambda line: line.split(',')) 
// MAGIC split_data_2 = t2.map(lambda line: line.split(',')) 
// MAGIC 
// MAGIC train = split_data_1.map(lambda line: LabeledPoint(int(line[54])-1,line[:54]))
// MAGIC test = split_data_2.map(lambda line: LabeledPoint(int(line[54])-1,line[:54]))
// MAGIC train.take(1)

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
// MAGIC from pyspark.mllib.util import MLUtils
// MAGIC 
// MAGIC maxDepth = [20,25,30]
// MAGIC #maxDepth = [2,3]
// MAGIC maxBins = [100]
// MAGIC #maxBins = [50]
// MAGIC numClasses = 7
// MAGIC categoricalFeaturesInfo = {x: 2 for x in range(11,54)}
// MAGIC impurity = "gini"
// MAGIC 
// MAGIC for md in maxDepth:
// MAGIC   for mb in maxBins:
// MAGIC     results = []
// MAGIC     # Train a DecisionTree model.
// MAGIC     #  Empty categoricalFeaturesInfo indicates all features are continuous.
// MAGIC 
// MAGIC     model = DecisionTree.trainClassifier(train, numClasses, categoricalFeaturesInfo, impurity, md, mb)
// MAGIC 
// MAGIC     # Evaluate model on test instances and compute test error
// MAGIC     #predictions = model.predict(test.map(lambda x: x.features))
// MAGIC     test_features  = test.map(lambda x: x.features)
// MAGIC     predictions = model.predict(test_features)
// MAGIC     labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
// MAGIC     testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(test.count())
// MAGIC     results += (md,mb,testErr)
// MAGIC     print('maxDepth: ', md, 'maxBins: ', mb, 'Test Error = ' + str(testErr))
// MAGIC     print('Learned classification tree model:')

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC #print(model.toDebugString())

// COMMAND ----------

// MAGIC %python
// MAGIC # Train a Random Forest model.
// MAGIC #  Empty categoricalFeaturesInfo indicates all features are continuous.
// MAGIC numTrees = [1,5,10,20,30]
// MAGIC numClasses = 7
// MAGIC categoricalFeaturesInfo =  {x: 2 for x in range(11,54)}
// MAGIC featureSubsetStrategy = "auto" # Let the algorithm choose.
// MAGIC impurity = "gini"
// MAGIC maxDepth = 10
// MAGIC maxBins = 100

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.mllib.tree import RandomForest
// MAGIC 
// MAGIC rmodel = RandomForest.trainClassifier(train, numClasses, categoricalFeaturesInfo,
// MAGIC   numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

// COMMAND ----------

// MAGIC %python
// MAGIC # Evaluate model on test instances and compute test error
// MAGIC test_features  = test.map(lambda x: x.features)
// MAGIC predictions = rmodel.predict(test_features)
// MAGIC labelsAndPredictions = test.map(lambda lp: lp.label).zip(predictions)
// MAGIC testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(test.count())
// MAGIC print('Test Error = ' + str(testErr))
// MAGIC print('Learned classification tree model:')
// MAGIC print(model.toDebugString())

// COMMAND ----------


