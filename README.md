# Viola-Jones_Algorithm
This is an implementation of the Viola-Jones Algorithm in python3. The results of the original algorithm are improved using Adaboost algorithm. 

The enviorment information 
Language = Python3
Packages 
numpy, open-cv, mathplotlib, os, joblib, multiprocessing, glob, re seaborn and pandas

One can install these by using the following commands 
numpy:   $ pip install numpy<br/>
open-cv: $ pip install opencv-python<br/>
mathplotlib: $ pip install mathplotlib<br/>
multiprocessing: $ pip install multiprocessing<br/>
joblib: $ pip install joblib<br/>
os: $ pip install os<br/>
pandas: $ pip install pandas<br/> 
glob $ pip install glob<br/>
seaborn $ pip install seaborn<br/>
re $ pip install re<br/>


One can use the sudo if having trouble or view the documentation online

# Operating system notes
If someone is running the code on a windows machine. They will have to change the path that is used in all of the file because the program was written a Unix based machine. The program makes used of parrellelization to decrease excecution time, so when running on single core machine on may observe longer ececution time.  

# Accessing Data Set
One can download the data set from git by running git lfs. Please refer to git lfs documentation. If one can on download the feature file, one can run the Haar_feature.py to extract their own feature file. 

# Files Information
Main.py used to plot the error extracting the features 
Displaying_features.py used to display the Haar features on a default image
ada_boost.py perfrom the adaboost algorthim on the data and ouputs the best features and their respective threshold to a file
Modified_Ada_boost.py same as adabosst but allow the user to place a gamma parameter to change the criteria.

