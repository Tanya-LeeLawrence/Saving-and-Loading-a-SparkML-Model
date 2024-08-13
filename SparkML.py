
# Import necessary libraries
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel

# Creating the spark session and context
sc = SparkContext()
spark = SparkSession \
    .builder \
    .appName("Saving and Loading a SparkML Model").getOrCreate()

# Create a DataFrame with sample data
mydata = [[46,2.5],[51,3.4],[54,4.4],[57,5.1],[60,5.6],[61,6.1],[63,6.4]]
columns = ["height", "weight"]
mydf = spark.createDataFrame(mydata, columns)
mydf.show()

# Convert data frame columns into feature vectors
assembler = VectorAssembler(
    inputCols=["height"],
    outputCol="features")
data = assembler.transform(mydf).select('features','weight')
data.show()

# Create and train the Linear Regression model
lr = LinearRegression(featuresCol='features', labelCol='weight', maxIter=100)
lr.setRegParam(0.1)
lrModel = lr.fit(data)

# Save the model
lrModel.save('infantheight2.model')

# Load the model
model = LinearRegressionModel.load('infantheight2.model')

# Predict function to predict the weight of an infant based on height
def predict(height):
    assembler = VectorAssembler(inputCols=["height"], outputCol="features")
    data = [[height, 0]]
    columns = ["height", "weight"]
    _ = spark.createDataFrame(data, columns)
    __ = assembler.transform(_).select('features', 'weight')
    predictions = model.transform(__)
    predictions.select('prediction').show()

# Predict the weight of an infant whose height is 70 CMs
predict(70)


# Save the model as `babyweightprediction.model`
lrModel.save('babyweightprediction.model')

# Load the model `babyweightprediction.model`
model = LinearRegressionModel.load('babyweightprediction.model')

# Predict the weight of an infant whose height is 50 CMs
predict(50)
