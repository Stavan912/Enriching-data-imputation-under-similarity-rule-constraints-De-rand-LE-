# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:31:19 2019

@author: abdul
"""
#%%libraries
import pandas as pd
import numpy as np
from math import sqrt
from tqdm import tqdm
import os
import glob
import timeit
#%%functions
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
        
    if enable_approx:
        if sample_data < len(dataset_mod):
            dataset_mod = dataset_mod.sample(n = sample_data, random_state = 42)
    
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
        if neighbour_df.iloc[dataset_mod.iloc[i].name].isnull().any():
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

#%%choices
dict_choice = {0: 'supervised', 1: 'unsupervised'}
choose = 0
choice = dict_choice[choose]


####enable approx = 1 when you want to sample the possible candidates for running on big datasets
###sample_data is for how many candidates you want to sample
enable_approx = 1
sample_data = 100

fill_mode = 'mode'
num_neighbours = 5

#%%main code

base_path = "C:\\Users\\abdul\\OneDrive\\Windsor\\Courses\\Data Mining\\"
sub_path = "Course Project Labeled Datasets\\Incomplete Datasets\\" if choose == 0 else "Course Project Datasets\\Incomplete Datasets Without Labels\\"

path = os.path.join(base_path, sub_path)

subfolders = os.listdir(path)

#numerical_subfolders = ['4-gauss', 'BCW', 'Bupa', 'CNP', 'DERM', 'Difdoug',
#'Glass',  'Ionosphere', 'Iris', 'PID', 'Sheart', 'Sonar', 'Spam', 'Letter',
# 'Wine', 'Yeast']

#numerical_subfolders = ['Spam']

#alphabet_subfolders = ['KDD']
alphabet_subfolders = ['TTTTEG', 'HOV', 'Abalone', 'MUSH', 'Splice', 'C4']

#string_subfolders = ['Aheart', 'CREDIT', 'Adult', 'KDD']
#string_subfolders = ['Aheart', 'CREDIT', 'Adult', 'KDD']

for subfolder in alphabet_subfolders:
    read_path = os.path.join(path, subfolder)
    csv_files = os.listdir(read_path)
    
    for csv in csv_files:
        
        write_dir = os.path.join("\\".join(path.split("\\")[:-2]), "Imputed Datasets", subfolder)
        
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
            
        write_path = os.path.join(write_dir, csv)
        
        if os.path.exists(write_path):
            continue
              
        start = timeit.default_timer()
        
        read_data = pd.read_excel(os.path.join(read_path, csv), header = None)
        data = read_data.copy()
        del read_data
        
        data, min_max = normalize(data)
        
        for column in data.columns:
            if data[column].isnull().all():
                data[column] = 0

        missing_values_dist = data.isnull().sum()
        
        missing_tuples = data[data.isnull().any(axis = 1)]
        
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
        
      
        stop = timeit.default_timer()
        
        try:
            data.to_excel(write_path, index = False, header = None)
        except:
            pass
        
        read_original_data = pd.read_excel(os.path.join("\\".join(path.split("\\")[:-2]), "Original Datasets",
                                                        subfolder + '.xlsx'), header = None)
        original_data = read_original_data.copy()
        del read_original_data
        
        if choice == 'supervised':
            original_data = original_data.drop(original_data.columns[-1],axis=1)
            data = data.drop(data.columns[-1], axis = 1)
        
        AE = ((original_data==data).sum()).sum() / original_data.size
        
        print(f"{csv.split('.xlsx')[0]}     AE: {AE}, Time Complexity: {round(stop - start, 2)}s")
        
#        NRMS_AE.loc[NRMS_AE["Datasets"] == csv.split('.xlsx')[0], "NRMS"] = NRMS
#        NRMS_AE.loc[NRMS_AE["Datasets"] == csv.split('.xlsx')[0], "AE"] = abs(AE)
#        NRMS_AE.loc[NRMS_AE["Datasets"] == csv.split('.xlsx')[0], "Time Complexity"] = f"{round(stop - start, 2)}s"
#        NRMS_AE.loc[NRMS_AE["Datasets"] == csv.split('.xlsx')[0], "Key Parameters"] = f"k = {num_neighbours}" 
        
#        try:
##            NRMS_AE.to_excel(f"NRMS_AE_{choice}.xlsx", index = False)
#        except:
#            pass