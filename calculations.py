from pyspark.sql import SparkSession, functions, types
import sys


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

def fill_null_with_mean(data, column_name):
    # Calculate mean for the column grouped by station
    column_avg = data.groupBy('station').agg(functions.avg(column_name).alias(f'avg_{column_name}'))

    # Join original data with station-wise mean
    data_with_avg = data.join(column_avg, 'station', 'left')

    # Fill null values in the column with station-specific averages
    filled_column = data_with_avg.withColumn(f'{column_name}_filled', functions.when(functions.col(column_name).isNull(), functions.col(f'avg_{column_name}')).otherwise(functions.col(column_name)))
    return filled_column

def main(in_directory, out_directory):
    # Read the JSON data into a DataFrame
    data = spark.read.json(in_directory, schema = observation_schema)

    # Fill null values for TMAX, TMIN, and PRCP columns with the averages of the respective stations values
    data = fill_null_with_mean(data, 'TMAX')
    data = fill_null_with_mean(data, 'TMIN')
    data = fill_null_with_mean(data, 'PRCP')

    # create a new column year that just conatin the year in whch the observation was recorded.
    data = data.withColumn('year', functions.year('date'))

    # Filter yearly data
    year_2016 = data.filter(functions.col("year") == 2016).select('station', 'city', 'Country', 'numVisitors', 'year', 'TMAX_filled', 'TMIN_filled', 'PRCP_filled')
    year_2017 = data.filter(functions.col("year") == 2017).select('station', 'city', 'Country', 'numVisitors', 'year', 'TMAX_filled', 'TMIN_filled', 'PRCP_filled')

    # Calculate yearly average TMAX, TMIN, PRCP for stations for 2016
    avg_tmax_2016 = year_2016.groupBy('station').agg(functions.avg('TMAX_filled').alias('avg_tmax_2016'))
    avg_tmin_2016 = year_2016.groupBy('station').agg(functions.avg('TMIN_filled').alias('avg_tmin_2016'))
    avg_prcp_2016 = year_2016.groupBy('station').agg(functions.avg('PRCP_filled').alias('avg_prcp_2016'))

    # join the average tmax, tmin and prcp data for the stations for year 2016
    avg_data = year_2016.join(avg_tmax_2016, 'station').join(avg_tmin_2016, 'station').join(avg_prcp_2016, 'station')
    # drop the columns not required
    avg_data = avg_data.drop(avg_data['TMAX_filled']).drop(avg_data['TMIN_filled']).drop(avg_data['PRCP_filled'])
    # get the rows with distinct values to make the dataset smaller
    distinct_avg_data = avg_data.select('station','city', 'Country', 'numVisitors','avg_tmax_2016', 'avg_tmin_2016', 'avg_prcp_2016').distinct()

    # Calculate yearly average TMAX, TMIN, PRCP for stations for 2017
    avg_tmax_2017 = year_2017.groupBy('station').agg(functions.avg('TMAX_filled').alias('avg_tmax_2017'))
    avg_tmin_2017 = year_2017.groupBy('station').agg(functions.avg('TMIN_filled').alias('avg_tmin_2017'))
    avg_prcp_2017 = year_2017.groupBy('station').agg(functions.avg('PRCP_filled').alias('avg_prcp_2017'))


    # join the average tmax, tmin and prcp data for the stations for year 2017
    avg_data_2017 = year_2017.join(avg_tmax_2017, 'station').join(avg_tmin_2017, 'station').join(avg_prcp_2017, 'station')
    # drop the columns not required
    avg_data_2017 = avg_data_2017.drop(avg_data_2017['TMAX_filled']).drop(avg_data_2017['TMIN_filled']).drop(avg_data_2017['PRCP_filled'])
    # get the rows with distinct values to make the dataset smaller
    distinct_avg_data_2017 = avg_data_2017.select('station','city', 'Country', 'numVisitors','avg_tmax_2017', 'avg_tmin_2017', 'avg_prcp_2017').distinct()

    # join the data for 2016 and 2017 to get the final dataset and save it in a json file
    all_data = distinct_avg_data.join(distinct_avg_data_2017, ['station', 'city','Country','numVisitors'])
    all_data.write.json(out_directory, mode='overwrite', compression='gzip')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    spark = SparkSession.builder.appName('AverageTMAX').getOrCreate()
    assert spark.version >= '3.2' # make sure we have Spark 3.2+
    spark.sparkContext.setLogLevel('WARN')

    main(in_directory, out_directory)