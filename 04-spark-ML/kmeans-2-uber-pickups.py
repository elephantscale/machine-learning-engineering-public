#!/usr/bin/env python
# coding: utf-8

# # Clustering : K-Means : Uber Pickups
# 
# This is data of Uber pickups in New York City.  
# The data is from this [kaggle competition](https://www.kaggle.com/fivethirtyeight/uber-pickups-in-new-york-city).
# 
# Sample data looks like this
# ```
# "Date_Time","Lat","Lon","Base"
# "4/1/2014 0:11:00",40.769,-73.9549,"B02512"
# "4/1/2014 0:17:00",40.7267,-74.0345,"B02512"
# "4/1/2014 0:21:00",40.7316,-73.9873,"B02512"
# "4/1/2014 0:28:00",40.7588,-73.9776,"B02512"
# ```

# ## Step 1: Load the Data
# We will also specify schema to reduce loading time

# In[2]:


# file to read

## sample file with 10,000 records
data_file="/data/uber-nyc/uber-sample-10k.csv"

## larger file with about 500k records
#data_file = "/data/uber-nyc/uber-raw-data-apr14.csv.gz"

## all data
# data_file = "/data/uber-nyc/raw/"


# In[3]:


from pyspark.sql.types import StringType, FloatType, StructField, StructType

pickup_time_field = StructField("pickup_time", StringType(), True)
lat_field = StructField("Lat", FloatType(), True)
lon_field = StructField("Lon", FloatType(), True)
base_field = StructField("Base", StringType(), True)

schema = StructType([pickup_time_field, lat_field, lon_field, base_field])


# In[4]:


get_ipython().run_cell_magic('time', '', 'uber_pickups = spark.read.option("header", "true").schema(schema).csv(data_file)')


# In[5]:


records_count_total = uber_pickups.count()
print("read {:,} records".format(records_count_total))
uber_pickups.printSchema()
uber_pickups.show(10)


# ## Step 2: Cleanup data
# make sure our data is clean

# In[6]:


uber_pickups_clean = uber_pickups.na.drop(subset=['Lat', 'Lon'])
records_count_clean = uber_pickups_clean.count()

print ("cleaned records {:,},  dropped {:,}".format(records_count_clean,  (records_count_total - records_count_clean)))


# ## Step 3 : Create Feature Vectors

# In[7]:


from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["Lat", "Lon"], outputCol="features")
featureVector = assembler.transform(uber_pickups_clean)
# featureVector.show()


# ## Step 4: Running Kmeans
# 
# Now it's time to run kmeans on the resultant dataframe.  We don't know what value of k to use, so let's just start with k=4.  This means we will cluster into four groups.
# 
# We will fit a model to the data, and then train it.

# In[8]:


from pyspark.ml.clustering import KMeans

num_clusters = 4
kmeans = KMeans().setK(num_clusters).setSeed(1)


# In[9]:


import time 

t1 = time.perf_counter()
model = kmeans.fit(featureVector)
t2 = time.perf_counter()


# In[10]:


wssse = model.computeCost(featureVector)
print ("k={},  wssse={},  time took {:,.2f} ms".format(k,wssse, ((t2-t1)*1000)))


# ## Step 6: Let's find the best K - Hyperparameter tuning
# 
# Let's try iterating and plotting over values of k, so we can practice using the elbow method.
# 

# In[11]:


import time 

kvals = []
wssses = []

# For lop to run over and over again.
for k in range(2,10):
    kmeans = KMeans().setK(k).setSeed(1)
    t1 = time.perf_counter()
    model = kmeans.fit(featureVector)
    t2 = time.perf_counter()
    wssse = model.computeCost(featureVector)
    print ("k={},  wssse={},  time took {:,.2f} ms".format(k,wssse, ((t2-t1)*1000)))
    kvals.append(k)
    wssses.append(wssse)


# In[12]:


import pandas as pd
df = pd.DataFrame({'k': kvals, 'wssse':wssses})
df


# ## Step 7 : Let's run K-Means with the best K we have choosen

# In[14]:


num_clusters = 6
kmeans = KMeans().setK(num_clusters).setSeed(1)

t1 = time.perf_counter()
model = kmeans.fit(featureVector)
t2 = time.perf_counter()

wssse = model.computeCost(featureVector)


print("Kmeans : {} clusters computed in {:,.2f} ms".format( num_clusters,  ((t2-t1)*1000)))
print ("num_clusters = {},  WSSSE = {:,}".format(num_clusters, wssse))


# In[15]:


t1 = time.perf_counter()
predicted = model.transform(featureVector)
t2 = time.perf_counter()

print ("{:,} records clustered in {:,.2f} ms".format(predicted.count(), ((t2-t1)*1000) ))

predicted.show()


# ## Step 8 : Print Cluster Center and Size

# In[16]:


cluster_count = predicted.groupby("prediction").count().orderBy("prediction")
cluster_count.show()
index = 0
for c in model.clusterCenters():
    print(index, c)
    index = index+1


# ## Step 9 : Ploting time!
# We are going to plot the results now.  
# Since we are dealing with GEO co-ordinates, let's use Google Maps!  
# 
# Go to the following URL :  
# [https://jsfiddle.net/sujee/omypetfu/](https://jsfiddle.net/sujee/omypetfu/)
# 
# - Run the code cell below
# - copy paste the output into Javascript section of the JSFiddle Editor (lower left)
# - and click 'Run'  (top nav bar)
# - Click on 'tidy' (top nav bar)  to cleanup code
# 
# See the following image 
# 
# <img src="../assets/images/kmeans_uber_trips_map.png" style="border: 5px solid grey ; max-width:100%;" />
# 
# You will be rewarded with a beautiful map of clusters on Google Maps
# 
# <img src="../assets/images/Kmeans_uber_trips.png" style="border: 5px solid grey ; max-width:100%;" />
# 
# Optional
# - You can 'fork' the snippet and keep tweaking

# In[17]:


### generate Javascript
s1 = "var clusters = {"

s2 = ""

prediction_count = predicted.groupby("prediction").count().orderBy("prediction").select("count").collect()
total_count = 0
cluster_centers = model.clusterCenters()
for i in range(0, num_clusters):
    count = prediction_count[i]["count"]
    lat = cluster_centers[i][0]
    lng = cluster_centers[i][1]
    total_count = total_count + count
    if (i > 0):
        s2 = s2 + ","
    s2 = s2 + " {}: {{ center: {{ lat: {}, lng: {} }}, count: {} }}".        format(i, lat, lng, count)
    #s2 = s2 + "{}: {{  center: {{ }}, }}".format(i)

s3 = s1 + s2 + "};"

s4 = """
function initMap() {
  // Create the map.
  var map = new google.maps.Map(document.getElementById('map'), {
    zoom: 10,
    center: {
      lat: 40.77274573,
      lng: -73.94
    },
    mapTypeId: 'roadmap'
  });

  // Construct the circle for each value in citymap.
  // Note: We scale the area of the circle based on the population.
  for (var cluster in clusters) {
    // Add the circle for this city to the map.
    var cityCircle = new google.maps.Circle({
      strokeColor: '#FF0000',
      strokeOpacity: 0.8,
      strokeWeight: 2,
      fillColor: '#FF0000',
      fillOpacity: 0.35,
      map: map,
      center: clusters[cluster].center,
"""

s5 = "radius: clusters[cluster].count / {} * 100 * 300 }});  }}}}".format(total_count)

# final
s = s3 + s4 + s5

print(s)


# ## Step 10 : Running the script
# 
# **Use the dowload script**
# 
# ```bash
# cd   ~/data/uber-nyc
# ./download-data.sh
# ```
# 
# This will download more data.
# 
# As we run on larger dataset, the execution will take longer and Jupyter notebook might time out.  So let's run this in command line / script mode
# 
# ```bash
# 
# $    cd   ~/ml-labs-spark-python/clustering
# 
# $    time  ~/spark/bin/spark-submit    --master local[*]  kmeans-uber.py 2> logs
# 
# ```
# 
# Watch the output
# 
