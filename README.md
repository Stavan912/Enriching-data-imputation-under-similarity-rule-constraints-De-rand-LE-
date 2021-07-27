# Enriching-data-imputation-under-similarity-rule-constraints-De-rand-LE-
The algorithm proposed here does imputation considering the similarity rule constraints which takes the differential dependencies into account. The differential dependencies empower the algorithm to perform well even if the data points have small variations. The issue of maximizing the missing data imputation is also evaluated by this algorithm.
1. download anaconda https://www.anaconda.com/distribution/

run the following commands in anaconda-prompt:

#create an virtual environment where you will install the necessary libraries and python
2. conda create --name rafi python=3.6

#activate the environment
3. activate rafi

#install the necessary libraries and packages

4. conda install pandas
5. conda install tqdm
6. pip install xlrd
7. conda install spyder
8. conda install openpyxl

#now whenever you want to access the environment open the anaconda prompt, activate your environment and open a python ide (i.e. spyder)

#run the code to do imputation for numerical datasets (change original_path and read_path as per your directory) 
numerical.py

#run the code to do imputation for categorial datasets
categorical.py

#run the code to do imputation for mixed datasets
mixed.py

These codes will save the imputed datasets in the same directory the code is saved. You just have to choose the dataset you want to
run your code on and the suffix of the dataset. 

#impute_everything_numerical.py #impute_everything_categorical.py  
It will impute everything and save the imputed datasets in your desired directory. 

#Calculate the NMSE and AE
calculate_nmse_ae.py
This code will calculate the NMSE and AE of your desired datasets and save everything in a csv file. 


Notes: 
1. The code successfully ran on all 44 files of 23 datasets. 
2. For KDD we were only able to check 2 files and for C4 we checked 21 files. 
3. Abalone will cause problem as there are some classes with only one tuple. 
4. Always make sure you have changed all the paths in the code as per your directory to run the code successfully
