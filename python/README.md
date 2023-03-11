[Paper](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) 


# Viola-Jones
This is a Python 3 implementation of the Viola-Jones Algorithm presented in the paper above. The algorithm uses tradition image features to find face in an image with small number of instructions. Making it perfect for low power and real time classification application. The application has been developed feature but the paper has been a fundamental for the fields of machine learning and computer vision.

# Environment Setup

> pip install -r requirements.txt

# Operating system notes

If someone is running the code on a windows machine. They will have to change the path that is used in all of the file because the program was written a Unix based machine. The program makes used of parallelization to decrease execution time, so when running on single core machine on may observe longer execution times than stated.  

# Creating Dataset

`python3 python/dataset_generation.py -d dataset/ -o .dataset/`

# Training the AdaBoost model

`python3 python/train.py -d .dataset/ -o results/ -n 100`

The results contain model in a json format including the decision stamp data and images of the features used to classify the model. 

# Testing the AdaBoost model

`python3 python/train.py -d .dataset/ -m results/100_round_model.json  -o results/`

# Files Information

`dataset_generation.py` used to generate the dataset csv from the the image dataset
`train.py` train the model for `n` number of rounds on a given dataset and save the model to a given output path.
`test.py` used to test the model on a test set and store the resulting statistics for a given dataset.
