// Databricks notebook source exported at Wed, 10 Feb 2016 20:28:09 UTC
// MAGIC %md
// MAGIC # Trees and Forests
// MAGIC #### Dataset Introduction
// MAGIC This is a popular dataset for classification. Given a feature vector of 14 census results, the problem is to predict whether a persons income is greater than 50K.  

// COMMAND ----------

// MAGIC %md
// MAGIC #### Open Files (use traindata to train, testdata to test)

// COMMAND ----------

sc.hadoopConfiguration.set("fs.s3n.awsAccessKeyId", "AKIAJH57T"+"SADMXPN"+"3NWA")
sc.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey", "cl7ON3wPVCf"+"a42eAzHjRD"+"v0iVJgsApuS"+"H3qwyMwF")

val trainFileRDD = sc.textFile("s3n://mlonspark/adult.traindata.numbers.csv")
val testFileRDD = sc.textFile("s3n://mlonspark/adult.testdata.numbers.csv")

trainFileRDD.take(10).foreach(println)
println
testFileRDD.take(10).foreach(println)


// COMMAND ----------

// MAGIC %md
// MAGIC #### Description of Fields
// MAGIC Note: For all categorial data, the number the number corresponds to a category. I.e. 1 = "Private", 2="Self-emp-not-inc" for the workclass (2nd) column.
// MAGIC 
// MAGIC * 0-age: continuous.
// MAGIC * 1-workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
// MAGIC * 2-fnlwgt: continuous.
// MAGIC * 3-education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
// MAGIC * 4-education-num: continuous.
// MAGIC * 5-marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
// MAGIC * 6-occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
// MAGIC * 7-relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
// MAGIC * 8-race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
// MAGIC * 9-sex: Female, Male.
// MAGIC * 10-capital-gain: continuous.
// MAGIC * 11-capital-loss: continuous.
// MAGIC * 12-hours-per-week: continuous.
// MAGIC * 13-native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
// MAGIC * 14-income: >50K, <=50K

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

// MAGIC %md
// MAGIC #### Create LabeledPoint RDD from the data (Both training and test datasets)

// COMMAND ----------

val trainlpRDD=trainFileRDD.map(_.split(",") match { case Array(age, workclass, fnlwgt, education, education_num, marital_status,occupation ,relationship, race ,sex ,capital_gain , capital_loss , hours_per_week , native_country, income ) =>LabeledPoint(income.toDouble-1.0, Vectors.dense(age.toDouble, workclass.toDouble-1.0, fnlwgt.toDouble, education.toDouble-1.0, education_num.toDouble, marital_status.toDouble-1.0,occupation.toDouble-1.0 ,relationship.toDouble-1.0, race.toDouble-1.0 ,sex.toDouble-1.0,capital_gain.toDouble , capital_loss.toDouble , hours_per_week.toDouble , native_country.toDouble-1.0) )})

val testlpRDD=trainFileRDD.map(_.split(",") match { case Array(age, workclass, fnlwgt, education, education_num, marital_status,occupation ,relationship, race ,sex ,capital_gain , capital_loss , hours_per_week , native_country, income ) =>LabeledPoint(income.toDouble-1.0, Vectors.dense(age.toDouble, workclass.toDouble-1.0, fnlwgt.toDouble, education.toDouble-1.0, education_num.toDouble, marital_status.toDouble-1.0,occupation.toDouble-1.0 ,relationship.toDouble-1.0, race.toDouble-1.0 ,sex.toDouble-1.0,capital_gain.toDouble , capital_loss.toDouble , hours_per_week.toDouble , native_country.toDouble-1.0) )})

// COMMAND ----------

// MAGIC %md
// MAGIC #### Train a Decision Tree
// MAGIC * Use gini impurity, maxdepth=5, maxBins=100
// MAGIC * Refer to here: http://spark.apache.org/docs/latest/mllib-decision-tree.html#classification
// MAGIC * for categoricalFeaturesInfo, you need to make it similar to Map\[Int, Int\]((1,9),(3,17)), this means the 1st column is categorial and has 9 different categories, and the 3rd column is categorial and has 17 different categories. Note: we start counting from 0. 

// COMMAND ----------

// Train a DecisionTree model.
//  Empty categoricalFeaturesInfo indicates all features are continuous.
val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]((1,9),(3,17),(5,9),(6,15),(7,7),(8,6),(9,3),(13,42))
val impurity = "gini"
val maxDepth = 5
val maxBins = 100

// COMMAND ----------

val model = DecisionTree.trainClassifier(trainlpRDD, numClasses, categoricalFeaturesInfo,
  impurity, maxDepth, maxBins)

// COMMAND ----------

// Evaluate model on test instances and compute test error
val labelAndPreds = testlpRDD.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testlpRDD.count()
println("Test Error = " + testErr)
println("Learned classification tree model:\n" + model.toDebugString)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Print the model to screen

// COMMAND ----------



// COMMAND ----------

// MAGIC %md
// MAGIC #### Calculate Test Error
// MAGIC I.e. (wrong predictions/total predictions)

// COMMAND ----------



// COMMAND ----------

// MAGIC %md
// MAGIC #### Train Random Forest and Calculate Error
// MAGIC Use 30 trees, maxdepth=10, maxBins=100

// COMMAND ----------



// COMMAND ----------

// MAGIC %md
// MAGIC #### Vary "Number of Trees" and "Max Tree Depth" and observe Error Rate
// MAGIC Train with Number of Trees, 1,5,10,20,30, and Max Tree Depth 2,3,5,10 to train the random forest model, and draw a plot showing how the error changes

// COMMAND ----------


