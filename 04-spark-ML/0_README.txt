README


In this directory, you will find
* Slides (SPARK-ML-webinar-gen.pdf)
* Kmeans jupyter notebook (kmeans-2-uber-pickups.ipynb) -- also same notebook in HTML format for quick viewing
* Kmeans python code (kmeans-2b-uber-full.py)
How to run the code:
You will need
* Spark installed
* And Jupyter notebook environment


Once you have those
* Open Jupyter notebook in Jupyter
* Adjust data paths (you can load directly from our S3 or GS public buckets)
* Hit Run-All-Cells
To run in command-line (for larger datasets)


On your local machine


$   time $SPARK_HOME/bin/spark-submit  --master local[*] --executor-memory 12g  kmeans-2b-uber-full.py 2> logs


* SPARK_HOME is where ever your spark is installed (mine is at :  ~/apps/spark/bin/spark-submit)
* 'Time' command measure how much times elapses for the whole program to operate


On Cloud (I am using Google Cloud)
* Create a Dataproc cluster
* SSH into master node (use the web SSH .. easy!)
* Upload the py file in here
   * On the cloud instance
$ nano  kmeans.py
   * Just copy-paste your content from your laptop 
   * Save and done :-) 
   * TODO : change data path to 
data_location = 'gs://elephantscale-public/data/uber-nyc/full2/'
   * Run as follows


$  time spark-submit --master yarn   kmeans.py  2> logs




Resources


Google Dataproc : https://cloud.google.com/dataproc


Spark 3 adds GPU support : https://www.infoworld.com/article/3543319/apache-spark-30-adds-nvidia-gpu-support-for-machine-learning.html


Kmeans demo : http://stanford.edu/class/ee103/visualizations/kmeans/kmeans.html 


Elephant Scale docker container for Spark + Jupyter + ML 
https://hub.docker.com/repository/docker/elephantscale/es-training