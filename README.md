[Paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) 

# Viola-Jones_Algorithm
This is an implementation of the Viola-Jones Algorithm in python3. The results of the original algorithm are improved using Adaboost algorithm.

# Enviornment Setup

> pip install -r requirements.txt

# Operating system notes

If someone is running the code on a windows machine. They will have to change the path that is used in all of the file because the program was written a Unix based machine. The program makes used of parallelization to decrease execution time, so when running on single core machine on may observe longer execution times than stated.  

# Accessing Data Set

One can download the data set from git by running git lfs. Please refer to git lfs documentation. If one can on download the feature file, one can run the Haar_feature.py to extract their own feature file. 

# Files Information

Main.py used to plot the error extracting the features 
Displaying_features.py used to display the Haar features on a default image
ada_boost.py perform the AdaBoost algorithm on the data and outputs the best features and their respective threshold to a file
Modified_Ada_boost.py same as AdaBoost but allow the user to place a gamma parameter to change the criteria.

