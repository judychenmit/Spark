// Databricks notebook source exported at Sat, 13 Feb 2016 00:25:40 UTC
// MAGIC %md
// MAGIC # Assignment Problem 1: Analyze Power Usage Data 

// COMMAND ----------

// MAGIC %md 
// MAGIC #### Instructions 
// MAGIC **This is an Open-ended problem. I'm not looking for one correct answer, but an organized analysis report on the data. **
// MAGIC 
// MAGIC We will use a dataset from one smartmeter to analyze the energy consumption pattern of a house. Analyze the electricity usage pattern and see what conclusion you can draw about the resident?
// MAGIC 
// MAGIC Note:
// MAGIC 
// MAGIC 1. You need to pay attention to missing data;
// MAGIC 2. calculate some aggregate values of energy usage and observe different type of trends (e.g. pattern in a day, pattern in a week, pattern in a year, etc);
// MAGIC 3. Use Spark to do your calculations, then use dataframes to draw some plots. Describe each plot briefly and draw some conclusions;
// MAGIC 4. You only need to use the simple Spark transformations and actions covered in class, no need to use machine learning methods yet. 

// COMMAND ----------

// MAGIC %md
// MAGIC #### How to work on and hand in this assignment
// MAGIC Simply clone this Notebook to your own directory. Write your analysis report, and send me the link (see address bar) of your Notebook before the assignment is due. 
// MAGIC 
// MAGIC If you prefer to do it in Python (or R), you can create a Python/R Notebook and send me the link to it. 

// COMMAND ----------

// MAGIC %md
// MAGIC #### Description of the Dataset
// MAGIC Source: https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption#
// MAGIC 
// MAGIC This archive contains 2075259 measurements gathered between December 2006 and November 2010 (47 months). 
// MAGIC 
// MAGIC Notes: 
// MAGIC 
// MAGIC 1.(global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3) represents the active energy consumed every minute (in watt hour) in the household by electrical equipment not measured in sub-meterings 1, 2 and 3. 
// MAGIC 
// MAGIC 2.The dataset contains some missing values in the measurements (nearly 1,25% of the rows). All calendar timestamps are present in the dataset but for some timestamps, the measurement values are missing: a missing value is represented by the absence of value between two consecutive semi-colon attribute separators. For instance, the dataset shows missing values on April 28, 2007.
// MAGIC 
// MAGIC 
// MAGIC Attribute Information:
// MAGIC 
// MAGIC 1.date: Date in format dd/mm/yyyy 
// MAGIC 
// MAGIC 2.time: time in format hh:mm:ss 
// MAGIC 
// MAGIC 3.global_active_power: household global minute-averaged active power (in kilowatt) 
// MAGIC 
// MAGIC 4.global_reactive_power: household global minute-averaged reactive power (in kilowatt) 
// MAGIC 
// MAGIC 5.voltage: minute-averaged voltage (in volt) 
// MAGIC 
// MAGIC 6.global_intensity: household global minute-averaged current intensity (in ampere) 
// MAGIC 
// MAGIC 7.sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered). 
// MAGIC 
// MAGIC 8.sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light. 
// MAGIC 
// MAGIC 9.sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.

// COMMAND ----------

// MAGIC %md
// MAGIC #### Load the data into an RDD

// COMMAND ----------

// MAGIC %python
// MAGIC # Set S3 
// MAGIC hadoopConf=sc._jsc.hadoopConfiguration()
// MAGIC hadoopConf.set("fs.s3n.awsAccessKeyId", "AKIAJH57T"+"SADMXPN"+"3NWA")
// MAGIC hadoopConf.set("fs.s3n.awsSecretAccessKey", "cl7ON3wPVCf"+"a42eAzHjRD"+"v0iVJgsApuS"+"H3qwyMwF")
// MAGIC 
// MAGIC 
// MAGIC # read txt file, gzipped files will be auto-unzipped
// MAGIC myRDD = sc.textFile("s3n://mlonspark/household_power_consumption.txt.gz")
// MAGIC 
// MAGIC print(myRDD.count())
// MAGIC print("\n".join(myRDD.take(10)))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Start Your Analysis Here

// COMMAND ----------

// MAGIC %python
// MAGIC split = myRDD.map(lambda s: s.split(';'))
// MAGIC subsplit = split
// MAGIC #subsplit = sc.parallelize(split.take(100000))
// MAGIC subsplit.take(1)

// COMMAND ----------

// MAGIC %python
// MAGIC subsplit = subsplit.filter(lambda s: s[2] != '?' and s[2] !='NaN' and s[2] !='0')

// COMMAND ----------

// MAGIC %python
// MAGIC import datetime
// MAGIC tuples = subsplit.map(lambda s: (datetime.datetime.strptime(s[0], '%d/%m/%Y').strftime('%Y%m%d'),float(s[2])))

// COMMAND ----------

// MAGIC %python
// MAGIC powerperday = tuples.reduceByKey(lambda a,b: (a+b)/60).sortByKey()
// MAGIC powerperday.take(5)

// COMMAND ----------

// MAGIC %python
// MAGIC tupleDF = sqlContext.createDataFrame(powerperday,('Date','Total Power Per Day'))
// MAGIC #print(tupleDF.collect()[6837:6838])
// MAGIC display(tupleDF)

// COMMAND ----------

// MAGIC %python
// MAGIC bydayofweek = powerperday.map(lambda s: (datetime.datetime.strptime(s[0], '%Y%m%d').weekday(),s[1]))
// MAGIC bydayofweek.take(10)
// MAGIC 
// MAGIC ## Monday is a 0 and Saturday is a 5!

// COMMAND ----------

// MAGIC %python
// MAGIC tupleDF = sqlContext.createDataFrame(bydayofweek,('Day','Total Power Per Day'))
// MAGIC display(tupleDF)

// COMMAND ----------

// MAGIC %md
// MAGIC Saturday seems to have the greatest average energy consumption by far.  It also exhibits the greatest variability of power consumption as well.

// COMMAND ----------

// MAGIC %python
// MAGIC display(tupleDF)

// COMMAND ----------

// MAGIC %md
// MAGIC Consumption on Saturday takes up 1/4 of all energy consumed with (curiously) an even split amongst the rest of the days.

// COMMAND ----------


