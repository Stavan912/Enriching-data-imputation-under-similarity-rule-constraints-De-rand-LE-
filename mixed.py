# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 01:12:24 2019

@author: abdul
"""
#%%
import pandas as pd
import numpy as np
from math import sqrt
from tqdm import tqdm
#%%

dataset_name = 'Aheart'
suffix = 'C_1'
choice = 'supervised'

fill_mode = 'mode'
num_neighbours = 5

#%%

###read_path tells you where to read the dataset
if choice == 'unsupervised':
    read_path = f"C:\\Users\\abdul\\OneDrive\\Windsor\\Courses\\Data Mining\\Course Project Datasets\\Incomplete Datasets Without Labels\\{dataset_name}\\{dataset_name}_{suffix}.xlsx"
elif choice == 'supervised':
    read_path = f"C:\\Users\\abdul\\OneDrive\\Windsor\\Courses\\Data Mining\\Course Project Labeled Datasets\\Incomplete Datasets\\{dataset_name}\\{dataset_name}_{suffix}.xlsx"

#path for from where to read the original data
original_path = f"C:\\Users\\abdul\\OneDrive\\Windsor\\Courses\\Data Mining\\Course Project Labeled Datasets\\Original Datasets\\{dataset_name}.xlsx"

####reads the data from the path
read_data = pd.read_excel(read_path, header = None)
data = read_data.copy()
del read_data
#%%functions that we will use for calculation

##normalizes each column of a given dataset depending on its minimum and maximum value
def normalize(df):
    result = df.copy()
    min_max = list()
    
    check_string = df.applymap(type).eq(str).any()
    
    if choice == 'supervised':
        check_string = check_string[:-1]
    
    for i, check in enumerate(check_string):
        if not check:
            max_value = df.iloc[:,i].max()
            min_value = df.iloc[:,i].min()
            result.iloc[:,i] = (df.iloc[:,i] - min_value) / (max_value - min_value)
            min_max.append((i, min_value, max_value))
            
    return result, min_max
            

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):

    distance = 0.0
    
    if choice == 'supervised':
        loop_iter = len(row1) - 1 
    else:
        loop_iter = len(row1)
    
    for i in range(loop_iter):
        if isinstance(row1[i], str):
            if row1[i] != row2[i]:
                distance += 1
        else:
            distance += abs(row1[i] - row2[i])
	
    return sqrt(distance)

###finds the nearest neigbours of a particular row belonging to the same class
def get_neighbours(dataset, concerned_row, num_neighbours):
    
    all_columns = concerned_row.index.to_list()
    concerned_row = concerned_row.dropna()
    full_columns = concerned_row.index.to_list()
    missing_columns = list(set(all_columns) - set(full_columns))
    
    dataset_mod = dataset.drop(columns = missing_columns)
    neighbour_df = dataset[missing_columns]
    
    if choice == 'supervised':
        dataset_mod = dataset_mod[dataset_mod.iloc[:,-1] == concerned_row.iloc[-1]]
    
    distances = list()
    for i, row in enumerate(dataset_mod.values):
        if dataset_mod.iloc[i].name == concerned_row.name:
            continue
        try:
            if np.isnan(row).any():
                continue
        except:
            if np.isnan(pd.isnull(np.array([row], dtype=object))).any():
                continue
        dist = euclidean_distance(concerned_row.values, row)
        distances.append((dataset_mod.iloc[i].name, row, dist))
        
    distances.sort(key=lambda x: x[2])
    
    neighbour_loc = list()
    
    if len(distances) < num_neighbours:
        loop_iter = len(distances)
    else:
        loop_iter = num_neighbours
    
    for i in range(loop_iter):
        neighbour_loc.append(distances[i][0])
        
    return neighbour_df.loc[neighbour_loc]

###will use if we impute categorical data as well
def check_string(dataset):
    
    checks = data.applymap(type).eq(str).any()
    
    for check in checks:
        if check:
            return True
        
    return False
#%%missing values
###normalize the data
#check_string = data.applymap(type).eq(str).any()
    
#for check in check_string:
#    if not check:
#        data, min_max = normalize(data)
#        break

data, min_max = normalize(data)

##find out the missing rows
missing_values_dist = data.isnull().sum()

missing_tuples = data[data.isnull().any(axis = 1)]

####generate the candidates based on eucliedean distance 
##differential dependency: rows must be from the same class
candidates = list()

for i,missing in tqdm(data.iterrows()):
    
    if not data.iloc[i].isnull().any():
        continue
    
    candidates = get_neighbours(data, missing, num_neighbours)
    
    columns = list()
    for column in candidates:
        if fill_mode == 'mean':
            data.loc[i, column] = candidates[column].mean()
        elif fill_mode == 'mode':
            data.loc[i, column] = candidates[column].mode().values[0]

####go back to original feature space by denormalizing the data
check_string = data.applymap(type).eq(str).any()
        
if choice == 'supervised':
    check_string = check_string[:-1]

counter = 0
for i, check in enumerate(check_string):
        if not check:
            min_value = min_max[counter][1]
            max_value = min_max[counter][2]
            data.iloc[:, min_max[counter][0]] = data.iloc[:, min_max[counter][0]] *(max_value - min_value) + min_value
            counter += 1
        
#######save the imputed data points in a csv file
#data.to_excel("output.xlsx", index = False, header = None)
    
###########read the original data
read_original_data = pd.read_excel(original_path, header = None)
original_data = read_original_data.copy()
del read_original_data

if choice == 'supervised':
    original_data = original_data.drop(original_data.columns[-1],axis=1)
    data = data.drop(data.columns[-1], axis = 1)

####calculate the NRMS 

AE = ((original_data==data).sum()).sum() / original_data.size


check_string = data.applymap(type).eq(str).any()
string_data = np.where(check_string)
numerical_data = np.where(check_string == 0)

numerical = data.iloc[: , numerical_data[0].tolist()].copy()
string = data.iloc[: , string_data[0].tolist()].copy()

original_numerical = original_data.iloc[: , numerical_data[0].tolist()].copy()
original_string = original_data.iloc[: , string_data[0].tolist()].copy()

RMS = (((original_numerical - numerical)**2).sum().sum())**0.5
NRMS = RMS / (((original_numerical**2).sum().sum())**0.5)

AE = ((original_string==string).sum()).sum() / original_string.size

print(f"NRMS: {NRMS} AE: {AE}")

#%%visualization

#hist = data.hist()

#plot histogram
'''
fig, axes = plt.subplots(nrows=3, ncols=1, figsize = (5, 10))

axes[0].set_title('Sepal length vs Petal length')
axes[0].hist2d(complete_tuples["Sepal length"].to_list(), complete_tuples["Petal length"].to_list(), bins=50)

axes[1].set_title('Sepal Width vs Sepal length')
axes[1].hist2d(complete_tuples["Sepal Width"].to_list(), complete_tuples["Sepal length"].to_list(), bins=50)

axes[2].set_title('Sepal Width vs Petal length')
axes[2].hist2d(complete_tuples["Sepal Width"].to_list(), complete_tuples["Petal length"].to_list(), bins=50)

#fig.tight_layout()
plt.show()
'''