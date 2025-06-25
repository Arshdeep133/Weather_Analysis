import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession, functions, types 

################# Spark schema for observations #############################################
observation_schema = types.StructType([
    types.StructField('station', types.StringType(), False),
    types.StructField('PRCP', types.FloatType(), False),
    types.StructField('TMAX', types.FloatType(), False),
    types.StructField('TMIN', types.FloatType(), False),
    types.StructField('date', types.DateType(), False),
    types.StructField('city', types.StringType(), False),
    types.StructField('Country', types.StringType(), False),
    types.StructField('numVisitors', types.IntegerType(), False),
    types.StructField('station-2', types.StringType(), False),
])

################# function to fill the null cells in the data with appropriate data #############################################
def fill_null_with_mean(data, column_name):
    # Calculate mean for the column grouped by station
    column_avg = data.groupBy('station').agg(functions.avg(column_name).alias(f'avg_{column_name}'))

    # Join original data with station-wise mean
    data_with_avg = data.join(column_avg, 'station', 'left')

    # Fill null values in the column with station-specific averages
    filled_column = data_with_avg.withColumn(f'{column_name}_filled', functions.when(functions.col(column_name).isNull(), functions.col(f'avg_{column_name}')).otherwise(functions.col(column_name)))
    return filled_column

def main(in_directory):
    # Read the JSON data into a DataFrame
    data = spark.read.json(in_directory, schema = observation_schema)

    # Fill null values for TMAX, TMIN, and PRCP columns
    data = fill_null_with_mean(data, 'TMAX')
    data = fill_null_with_mean(data, 'TMIN')
    data = fill_null_with_mean(data, 'PRCP')

    data = data.withColumn('year', functions.year('date'))
   

    # Filter yearly data
    year_2016 = data.filter(functions.col("year") == 2016).select('station', 'city', 'Country', 'numVisitors', 'year', 'TMAX_filled', 'TMIN_filled', 'PRCP_filled')
    year_2017 = data.filter(functions.col("year") == 2017).select('station', 'city', 'Country', 'numVisitors', 'year', 'TMAX_filled', 'TMIN_filled', 'PRCP_filled')

    # convert the spark dataframe to pandas dataframe and drop the columns with null values to carry out the machine learning 
    pd_2016 = year_2016.toPandas()
    pd_2016.dropna(inplace=True)

    pd_2017 = year_2017.toPandas()
    pd_2017.dropna(inplace=True)

    # values to train the machine learning model on
    features = pd_2016[['TMAX_filled', 'TMIN_filled', 'PRCP_filled']]
    target = pd_2016['numVisitors']

    # splitting data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # machine learning model
    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(n_estimators=100, max_depth=15)
    )

    # train the data
    model.fit(X_train, y_train)

    # printing the machine learning model score on the terminal
    print("The score of Random forest regressor",model.score(X_test, y_test))

    # the unlabelled data for which the model will make predictions for
    X_unlabelled = pd_2017[['TMAX_filled', 'TMIN_filled', 'PRCP_filled']]

    # predicting the values of num of visitors for X_unlablled and saving the result in pandas dataframe
    predicted_labelled = pd.DataFrame({'city': pd_2017['city'], 'tmax': pd_2017['TMAX_filled'], 'tmin': pd_2017['TMIN_filled'], 'prp': pd_2017['PRCP_filled'] ,'truth': pd_2017['numVisitors'], 'prediction': model.predict(X_unlabelled)})
    # saving the result in mlPredictions files
    predicted_labelled.to_csv('mlPredictions')
 


if __name__=='__main__':
    in_directory = sys.argv[1]
    spark = SparkSession.builder.appName('extra').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory)
