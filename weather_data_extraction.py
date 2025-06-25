import sys
from pyspark.sql import SparkSession, functions, types, Row



spark = SparkSession.builder.appName('GHCN extracter').getOrCreate()

ghcn_path = '/courses/datasets/ghcn-splits'
#ghcn_path ='weather-1'
output = 'ghcn-output'

################# Spark schema for observations #############################################
observation_schema = types.StructType([
	types.StructField('station', types.StringType(), False),
	types.StructField('date', types.StringType(), False),  # becomes a types.DateType in the output
	types.StructField('observation', types.StringType(), False),
	types.StructField('value', types.IntegerType(), False),
	types.StructField('mflag', types.StringType(), False),
	types.StructField('qflag', types.StringType(), False),
	types.StructField('sflag', types.StringType(), False),
	types.StructField('obstime', types.StringType(), False),
])

################# Spark schema for stations data available in station_cities.csv############
station_schema = types.StructType([
	types.StructField('city', types.StringType(), False),
	types.StructField('Country', types.StringType(), False),
	types.StructField('numVisitors', types.IntegerType(), False),
	types.StructField('station', types.StringType(), False),
	types.StructField('station-2', types.StringType(), False),
])



def main(in_directory):
    
	# read the station data from csv with station_schema
	station = spark.read.csv(in_directory, header=None, schema=station_schema)
	# Only keep the cities from top 100 tourist destinations whose stationID data is available with us 
	station = station.filter(station['station'] != 'NULL')


	# read all the observations data from the folder with observation_schema
	obs = spark.read.csv(ghcn_path, header=None, schema=observation_schema)
	# keep only some years: still a string comparison here
	obs = obs.filter((obs['date'] >= '2016') & (obs['date'] <= '2018')) #2018 doesn't have data when we looked at the output
	# filter the data to keep the valid data for TMAX, PRCP, TMIN
	obs = obs.filter(functions.isnull(obs['qflag'])).cache()
	obs = obs.drop(obs['mflag']).drop(obs['qflag']).drop(obs['sflag']).drop(obs['obstime'])
	obs = obs.filter(obs['observation'].isin('TMAX', 'PRCP','TMIN'))

	# group the data based on station ID and data, then transform the data from long to wide format 
	obs = obs.groupBy('station', 'date') \
        .pivot('observation') \
        .agg(functions.first('value'))

	# parse the date string into a real date object    
	obs = obs.withColumn('newdate', functions.to_date(obs['date'], 'yyyyMMdd'))
	obs = obs.drop('date').withColumnRenamed('newdate', 'date')
    
	# Join the resulting obs with stations to get the city, country and number of visitors in the obs dataframe
	obs = obs.join(functions.broadcast(station), on='station')

	# drop the columns not required by us
	obs = obs.drop('station-2')

	# save the resultig dataframe on a json file 
	obs.write.json(output, mode='overwrite', compression='gzip')
    

if __name__=='__main__':
	in_directory = sys.argv[1]
	main(in_directory)
