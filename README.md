# Weather Analysis

Our project is focused on studying the correlation between the number of tourists visiting a city and its weather conditions. We have employed different stages such as data extraction, filtering, analysis, visualization, hypothesis testing, and machine learning techniques to draw insights and predict the number of tourists based on weather data. For this study, we have concentrated on GHCN data available on the cluster for the years 2016 and 2017, and the top 100 tourist destinations available on Wikipedia for 2018. We have chosen the tourist data for 2018 as it has the highest number of cities with available weather data. 

# Sequence of running the files
1. weather_data_extraction.py
2. calculations.py
3. testing.py
4. machineLearning.py

# Workflow:


# Data Extraction: 

We have extracted data from Wikipedia and manually identified the valid station IDs for the corresponding cities. You can find the station IDs, cities and country names along with number of visitors in the 'station_cities.csv' file. 



# Data Preprocessing: 

We began by uploading the 'station_cities.csv' file to the cluster using the following commands: 

- From the local terminal: 'scp station_cities.csv cluster.cs.sfu.ca:' 

- From the gateway terminal: 'hdfs dfs -copyFromLocal station_cities.csv hdfs://controller.local:54310/user/<userid>/'

To upload weather_data_extraction.py to cluster run following command from local terminal: 'scp weather_data_extraction.py cluster.cs.sfu.ca:'
 
Once the data and python file is available on the cluster, we ran the following command on the remote terminal: 'spark-submit weather_data_extraction.py station_cities.csv'. This command generated an output file called 'ghcn-output', which has the required data subset for tourist sites for the years 2016 and 2017 for further analysis. 


To copy the output file from the cluster to the local computer, we used the following commands: 

- From the remote terminal: 'hdfs dfs -get ghcn-output' 

- From the local terminal: 'scp -r cluster.cs.sfu.ca:ghcn-output /home/<userid>/sfuhome/CMPT353/project' 


Note that '/home/<userid>/sfuhome/CMPT353/project' is the path where we stored the folder on our local computer. You can change it to any other desired location. 

 

# Extraction and Calculations: 

Once you have this folder on your local computer, you can start the data extraction process, which will give us the data in column format, which is required for us to analyze the data further.  

To do this, run calculations.py using the following spark command locally:  

spark-submit calculations.py ghcn-output calcOut  

This will produce an output folder: calcOut, with a json.gz file. Copy that to the main project folder and rename it avgData.json.gz, as pandas require a single file as input to run the program, not a folder. 

 
 
# Hypothesis Testing: 

After this, run testing.py using the command: 

python3 testing.py avgData.json.gz 

This will produce the t-results in txt file named testing_outputs.txt and output_plot.png with three plots that display the Data, best-fit line, and Lowess smoothened line for avgTMax, avgTMin, and avgPrecipitation data from the 75 cities for the years 2016 and 2017. 

 
 
# Machine Learning: 

From the above t-test's p-value, we can assume a relationship between the number of visitors and weather conditions. We wanted to check if we could predict the number of tourists in a city based on the temperature and precipitation data, for which we used machine learning. We used the data for 2016 to train and validate the machine learning model, and we used weather data for 2017 to predict the number of visitors. For machine learning analysis, run the file machineLearning.py using the below spark command locally: 

spark-submit machineLearning.py ghcn-output  

This will produce a csv file named: mlPredictions.csv which shows the predicted and actual number of visitors for the cities for year 2017. 
 
 
# Provided files: 

We have provided the following files in the git repository for your ease: 

- ghcn-output folder in the repository has weather and tourist data. 

- The calcOut folder has extracted focused data for 2016 and 2017, along with average TMAX, TMIN, and PRCP. After all the extraction and filtering, only 75 cities are left. 

- avgData.json.gz file is a copy of data in the CalcOut folder used in testing.py 

- output_plot.png: shows the plots we got from running testing.py 

- out2017: This folder contains predictions made using the machine learning model on 2017 data. 

# Libraries Used:
1. sys
2. pandas
3. numpy
4. scikit-learn (sklearn)
5. pyspark
6. scipy
7. matplotlib
8. statsmodels

To install any of the above libaries you can use pip install.

# Results:  

The analysis provided valuable insights into the correlation between weather conditions and the number of visitors in the city. The machine learning models showed a certain degree of accuracy in predicting the tourist traffic based on weather data. 


# Contributors  

Arshdeep Kaur and Jasleen Kaur 

 
