from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum
import pandas as pd
import seaborn as sns
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from itertools import combinations
from pyspark.sql.functions import corr
import matplotlib.pyplot as plt


spark = SparkSession.builder.appName("myexample").getOrCreate()
df = spark.read.csv(r"C:\Users\METACO\Desktop\ML_hw_dataset.csv", header=True, inferSchema=True)
df.show()


#nulls
null_counts = df.select([sum(col(column).isNull().cast("int")).alias(column) for column in df.columns])
has_nulls = any(null_counts.collect()[0])
if not has_nulls:
    print("no nulls detected.")


# df to numeric_df
string_cols = [c for c,d in df.dtypes if d == "string"]
indexers = []
for col_name in string_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name+"_index")
    indexers.append(indexer)
pipeline = Pipeline(stages=indexers)
numeric_df = pipeline.fit(df).transform(df)
numeric_df = numeric_df.drop(*string_cols)
numeric_df.show()

#Correlation show
mycol = "corr_features"
assembler = VectorAssembler(inputCols=numeric_df.columns, outputCol=mycol)
cache = assembler.transform(numeric_df).select(mycol)
matrix = Correlation.corr(cache, mycol).collect()[0][0] 
mymatrix = matrix.toArray().tolist() 
print(type(mymatrix))
sns.heatmap(mymatrix, cmap='viridis', annot=True)
plt.show()


#correlation:
feature_cols = [c for c in numeric_df.columns if c != "y"]
label_indexer = StringIndexer(inputCol="y", outputCol="label")
numeric_df = label_indexer.fit(numeric_df).transform(numeric_df)
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
numeric_df = assembler.transform(numeric_df)
corr_matrix = Correlation.corr(numeric_df, "features").head()
corr_array = corr_matrix[0].toArray()
corr_list = []
for i, col_name in enumerate(feature_cols):
    corr_value = corr_array[i][-1]
    corr_list.append((col_name, corr_value))
corr_list.sort(key=lambda x: abs(x[1]), reverse=True)
top_features = [c[0] for c in corr_list[:16]]
selected_cols = top_features + ["y"]
df_correlation = numeric_df.select(*selected_cols)
df_correlation.show()

#lr before
train_df, test_df = numeric_df.randomSplit([0.8, 0.2], seed=42)
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_df)
predictions = lr_model.transform(test_df)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
auc = evaluator.evaluate(predictions)
print("Accuracy on Numeric_df and 20 columns= {:.10f}".format(auc))


#lr after
feature_cols = [c for c in df_correlation.columns if c != "y"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_correlation = assembler.transform(df_correlation).select(col("features"), col("y").alias("label"))
train_df, test_df = df_correlation.randomSplit([0.8, 0.2], seed=42)
lr = LogisticRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_df)
predictions = lr_model.transform(test_df)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
auc = evaluator.evaluate(predictions)
print("Accuracy on FinalDF on 16 columns= {:.10f}".format(auc))
