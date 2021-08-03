from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# training = spark.read.option("header", True).option("inferSchema", True).csv("./spark.csv")
training = spark.read.csv(path="./spark.csv", header=True, inferSchema=True)
training.show()

col = ["id", "text", "label"]
training = training.select(col)

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)

pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
model = pipeline.fit(training)

model.write().overwrite().save('model')
# model = PipelineModel.load("model")

print("OK")
