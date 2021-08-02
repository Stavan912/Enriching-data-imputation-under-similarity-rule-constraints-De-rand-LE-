# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:15:40 2019

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

#%%
imputed_dataset_path = "C:\\Users\\abdul\\OneDrive\\Windsor\\Courses\\Data Mining\\Course Project Labeled Datasets\\Imputed Datasets\\"

files = os.listdir(imputed_dataset_path)

original_dataset_path = "C:\\Users\\abdul\\OneDrive\\Windsor\\Courses\\Data Mining\\Course Project Labeled Datasets\\Original Datasets\\"

numerical_subfolders = ['4-gauss', 'BCW', 'Bupa', 'CNP', 'DERM', 'Difdoug',
'Glass',  'Ionosphere', 'Iris', 'PID', 'Sheart', 'Sonar', 'Spam',
 'Wine', 'Yeast']

alphabet_subfolders = ['TTTTEG', 'HOV', 'MUSH', 'Splice', 'C4']
mixed_subfolders = ['Aheart', 'CREDIT', 'Adult', 'KDD', 'Abalone', 'Letter']

a = ['Bupa', 'TTTTEG', 'CREDIT']

nrms_ae_path = "C:\\Users\\abdul\\OneDrive\\Windsor\\Courses\\Data Mining\\Course Project Labeled Datasets\\Table-NRMS-AE.xlsx"
nrms_ae = pd.read_excel(nrms_ae_path, header = None)
nrms_ae = nrms_ae[1:]
nrms_ae.columns = ["Datasets", "NRMS", "AE", "Key Parameters"]

final_result = pd.DataFrame({'Datasets' : files, 'NRMS': float('NaN'), 'AE' : float('NaN'), 'key_parameters': 'k = 5'})

###choose numerical, alphabet or mixed subfolders to calculate NRMS, AE or NRMS+AE
for file in numerical_subfolders:
    dataset_name = file
    
    path_original = os.path.join(original_dataset_path, file + '.xlsx')
    
    read_original = pd.read_excel(path_original, header = None)
    original = read_original.copy()
    del read_original
    
#    original = original.drop(original.columns[-1],axis=1)
    
    path = os.path.join(imputed_dataset_path, file)
    subpaths = os.listdir(path)
    
    ae_mean = list() 
    nrms_mean = list()
    
    for subpath in subpaths:
        
        read_path = os.path.join(path, subpath)
        read_data = pd.read_excel(read_path, header = None)
        data = read_data.copy()
        del read_data
        
#        data = data.drop(data.columns[-1], axis = 1)

        
        
        
        if file in alphabet_subfolders:
            AE = ((original==data).sum()).sum() / original.size
            ae_mean.append(AE)
            nrms_ae.loc[nrms_ae["Datasets"] == subpath.split('.xlsx')[0], "AE"] = AE
            nrms_ae.loc[nrms_ae["Datasets"] == subpath.split('.xlsx')[0], "Key Parameters"] = "k = 5" 
            
        if file in mixed_subfolders:
            
            check_string = data.applymap(type).eq(str).any()
            string_data = np.where(check_string)
            numerical_data = np.where(check_string == 0)
            
            numerical = data.iloc[: , numerical_data[0].tolist()].copy()
            string = data.iloc[: , string_data[0].tolist()].copy()
            
            original_numerical = original.iloc[: , numerical_data[0].tolist()].copy()
            original_string = original.iloc[: , string_data[0].tolist()].copy()
            
            RMS = (((original_numerical - numerical)**2).sum().sum())**0.5
            NRMS = RMS / (((original_numerical**2).sum().sum())**0.5)
            nrms_mean.append(NRMS)
            nrms_ae.loc[nrms_ae["Datasets"] == subpath.split('.xlsx')[0], "NRMS"] = NRMS
            
            AE = ((original_string==string).sum()).sum() / original_string.size
            ae_mean.append(AE)
            nrms_ae.loc[nrms_ae["Datasets"] == subpath.split('.xlsx')[0], "AE"] = AE
            nrms_ae.loc[nrms_ae["Datasets"] == subpath.split('.xlsx')[0], "Key Parameters"] = "k = 5" 
            
        if file in numerical_subfolders:
            RMS = (((original - data)**2).sum().sum())**0.5
            NRMS = RMS / (((original**2).sum().sum())**0.5)
            nrms_mean.append(NRMS)
            
            
            
            nrms_ae.loc[nrms_ae["Datasets"] == subpath.split('.xlsx')[0], "NRMS"] = NRMS
            nrms_ae.loc[nrms_ae["Datasets"] == subpath.split('.xlsx')[0], "Key Parameters"] = "k = 5" 
        
        try:
            print(f"{subpath} NRMS: {NRMS}")
            print(f"{subpath} AE: {AE}")
        except:
            pass
    ae_mean = np.array(ae_mean)
    mean_ae = ae_mean.mean()
    
    nrms_mean = np.array(nrms_mean)
    mean_nrms = nrms_mean.mean()
    
    final_result.loc[final_result["Datasets"] == file, "NRMS"] = mean_nrms
    final_result.loc[final_result["Datasets"] == file, "AE"] = mean_ae
    
try:
    nrms_ae.to_excel(f"NRMS_AE.xlsx", index = False)
except:
    pass
    
        
        