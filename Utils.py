#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:03:32 2020

@author: sai
"""
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV
from math import factorial
import genTriplets
import psutil
from sys import getsizeof
import pickle

RSEED=42
gb = pow(10,9)

def nPr(n, r):
    return int(factorial(n)/factorial(n-r))

def CLTextFileReader(data_dir, fname, conv_field='state'):
    df = pd.read_csv(data_dir+fname, sep='\t', converters={conv_field: lambda x: str(x)})
    return df, len(df)

def fine_tune_xgb(model, train, train_labels, test, test_labels):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 30, stop = 60, num = 6)] # Fine tuning 1
    # n_estimators = [int(x) for x in np.linspace(start = 700, stop = 1500, num = 10)] # Fine tuning 2
    # n_estimators = [int(x) for x in np.linspace(start = 1000, stop = 1100, num = 5)]
    # Number of features to consider at every split
    #max_features = ['auto', 'sqrt'] fine tuning 1
    
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(30, 60, num = 6)] # Fine tuning 1
    # max_depth.append(None) # Fine tuning 1
    # max_depth = [int(x) for x in np.linspace(50, 100, num = 5)]
    
    # Minimum number of samples required to split a node
    lr = [0.30, 0.35, 0.4, 0.5, 0.55] # Fine tuning 1
    
    # Alpha
    gamma = [x for x in np.linspace(0.10, 0.70, num = 15)] # Fine tuning 1
    
    # Objective function
    #obj =['reg:logistic', 'multi:softprob', 'binary:logistic']
    
    random_grid = {'n_estimators': n_estimators,
                    'learning_rate': lr,
                    'max_depth': max_depth,
                    'gamma': gamma}

    # Use the random grid to search for best hyperparameters
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
    search = rf_random.fit(train, train_labels)

    print(' Best parameters for the model ')
    print(search.best_params_)

    fine_tuned_model = rf_random.best_estimator_
    # fine_tuned_model = RandomForestClassifier(n_estimators=100, 
    #                            random_state=RSEED, 
    #                            max_features = 'sqrt',
    #                            max_depth = 55,
    #                            min_samples_leaf = 1,
    #                            min_samples_split = 10,
    #                            n_jobs=-1, verbose = 1)
    # fine_tuned_model.fit(train, train_labels)
    
    # Testing Fine-tuned predictions (to determine performance)
    ft_rf_predictions = fine_tuned_model.predict(test)
    ft_rf_probs = fine_tuned_model.predict_proba(test)[:, 1]
    
    ft_recall = recall_score(test_labels, ft_rf_predictions, average='weighted')
    ft_precision = precision_score(test_labels, ft_rf_predictions, average='weighted')
    print('Fine tuned test recall', ft_recall)
    print('Fine tuned test precision',ft_precision)


def splitTrain(tcodes, tclasses, indices, tsize=0.2):
    #20% examples in test/eval data
    train, test, train_labels, test_labels, train_idx, test_idx = train_test_split(tcodes,
                                          tclasses,
                                          indices,
                                          stratify = tclasses,
                                          test_size = tsize, 
                                          random_state = RSEED)
    return train, test, train_labels, test_labels, train_idx, test_idx

def getDataLoader(data_dir, csv_file, code_map, cols=[], dim=602, encoding='barcode'):
    print(' Creating triplet vectors ...', len(code_map))
    return getData(csv_file, code_map, cols=cols, dim=dim,encoding=encoding)

def getData(debug_csv_file, code_map, cols=[], dim=602, encoding='barcode'):
    train=[]
    train_labels=[]
    train_idx=[]
    cmap={}
    cache_map={}
    if(len(cols)==0):
        cols=[1,2,3,4,5]
    
    print(' Using ', encoding, ' as feature representation ..')
    with open(debug_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        idx=-1 # 0-index
        prev_dreamid=1
        print('** Processing colony ', prev_dreamid)
        cmap = code_map[prev_dreamid]
        eval_flag=False
        
        for row in csv_reader:
            #Skip the header
            if(idx<0):
                idx+=1
                continue
            
            # colony id
            dreamid = int(row[cols[0]])
            
            if(len(cache_map)>1000):
                cache_map={}
            
            # Get the code map for the new colony
            if(prev_dreamid!=dreamid):
                print(' Processing colony ', dreamid)
                cmap = code_map[dreamid]
                prev_dreamid=dreamid
            
            # Read the row - triplet and class details
            tc1 = row[cols[1]]
            tc2 = row[cols[2]]
            tc3 = row[cols[3]]
            #cl_lbl = int(row[cols[4]])-1 # 0-index labels Labels =[0,3]
            if(len(cols)==5):
                cl_lbl = int(row[cols[4]]) # 1-index labels Labels =(1,2,3)
            else:
                cl_lbl=-1
            
            triplet_hvec=[]
            
            # For barcode
            if(encoding=='barcode'):
                triplet_code = genTriplets.concateVector(cmap[tc1], cmap[tc2], cmap[tc3])
                triplet_vec = genTriplets.encodeVector(triplet_code, dim)
                triplet_vec=triplet_vec.squeeze()
                train.append(triplet_vec) #For Barcode Implementation
            #For Hamming code
            elif(encoding=='hammingcode' or encoding=='hybrid'):
                triplet_hvec = genTriplets.hammingVector(cmap[tc1], cmap[tc2], cmap[tc3], 
                                                    (tc1,tc2,tc3),
                                                    cache_map)
                if(encoding=='hybrid'):
                    triplet_comb = getCombinedVec(triplet_vec, triplet_hvec)
                    train.append(triplet_comb)
                else:
                    #print(triplet_hvec.shape)
                    train.append(triplet_hvec)
            
            train_labels.append(cl_lbl)
            train_idx.append(idx)
            # increment the index
            idx+=1
    
    code_map=None #GC
    return train, train_labels, train_idx


def getCombinedVec(barcode, hammingcode):
    cvec = np.concatenate((barcode[:11],hammingcode[:11],
                           barcode[11:22],hammingcode[11:22],
                           barcode[22:32],
                           np.array([genTriplets.DELIM], dtype=np.int8()),
                           hammingcode[22:32]))

    return cvec
def getMetrics(labels, predictions, average='weighted'):
    recall = recall_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    #roc = roc_auc_score(labels, predictions, average='micro')
    return recall, precision

def saveModel(model, mfname):
    # Save the model into pickle file
    pickle.dump(model, open(mfname, "wb"))
    
def getModel(mfname):
    return pickle.load(open(mfname, "rb"))

def write_output(out_writer, full_data_info, idx, predictions, probs, ignoreIndexCol=True):
    
    print(' Writing ',len(full_data_info), ' triplets into file ...', len(idx))
    for i in range(len(idx)):
        if(ignoreIndexCol):
            row = [full_data_info.dreamID[idx[i]], 
              full_data_info.barcode[idx[i]],
              full_data_info.triplet_cell1[idx[i]],
              full_data_info.triplet_cell2[idx[i]],
              full_data_info.triplet_cell3[idx[i]],
              full_data_info['class'][idx[i]],
              predictions[i], 
              probs[i][predictions[i]-1],
              probs[i][0], probs[i][1], probs[i][2]]
        else:
            row = [idx[i], 
              full_data_info.dreamID[idx[i]], 
              full_data_info.barcode[idx[i]],
              full_data_info.triplet_cell1[idx[i]],
              full_data_info.triplet_cell2[idx[i]],
              full_data_info.triplet_cell3[idx[i]],
              full_data_info['class'][idx[i]],
              predictions[i], 
              probs[i][predictions[i]-1],
              probs[i][0], probs[i][1], probs[i][2]]
        out_writer.writerow(row)
        
def write_test_output(out_writer, full_data_info, idx, predictions, probs):
    for i in range(len(idx)):
        row = [full_data_info.dreamID[idx[i]], 
              full_data_info.barcode[idx[i]],
              full_data_info.triplet_cell1[idx[i]],
              full_data_info.triplet_cell2[idx[i]],
              full_data_info.triplet_cell3[idx[i]],
              predictions[i], 
              probs[i][predictions[i]-1],
              probs[i][0], probs[i][1], probs[i][2]]
        out_writer.writerow(row)
        
        
def buildDictFromDF(df, kcol, vcol):
    dict={}
    for idx in range(0, len(df)):
        dict[df[kcol].iloc[idx]]=df[vcol].iloc[idx]
    return dict        

def buildCodeMap(data_dir):
    code_map={}
    for idx in range(1,101):
        cid = str(idx).zfill(4)
        csv_file = data_dir+'SubC2_train_TXT/SubC2_train_'+cid+'.txt'
        # Read the cell-barcode map csv file
        cell_barcode_df = pd.read_csv(csv_file, sep='\t', header=None)
        #cell_barcode_map = cell_barcode_df.to_dict()
        cell_barcode_map = buildDictFromDF(cell_barcode_df, 0, 1)
        code_map[idx]=cell_barcode_map
        cell_barcode_df=None
    return code_map

def build_SC1_CodeMap(data_dir, dslen):
    code_map={}
    for idx in range(1,dslen+1):
        cid = str(idx)
        csv_file = data_dir+'sc1_train_txt/sub1_train_'+cid+'.txt'
        # Read the cell-barcode map csv file
        cell_barcode_df = pd.read_csv(csv_file, sep='\t', header=None)
        #cell_barcode_map = cell_barcode_df.to_dict()
        cell_barcode_map = buildDictFromDF(cell_barcode_df, 0, 1)
        code_map[idx]=cell_barcode_map
        cell_barcode_df=None
    return code_map

def build_SC1_TestCodeMap(data_dir, dslen):
    code_map={}
    for idx in range(1,dslen+1):
        cid = str(idx)
        csv_file = data_dir+'sc1_test_txt/sub1_test_'+cid+'.txt'
        # Read the cell-barcode map csv file
        cell_barcode_df = pd.read_csv(csv_file, sep='\t', header=None)
        #cell_barcode_map = cell_barcode_df.to_dict()
        #print(cid,"\t",len(cell_barcode_df))
        cell_barcode_map = buildDictFromDF(cell_barcode_df, 0, 1)
        code_map[idx]=cell_barcode_map
        cell_barcode_df=None
    return code_map


def printMemInfo(train_data, eval_data, labels=[],  eval_labels=[]):
    process = psutil.Process()
    
    train_mem = getsizeof(train_data)
    if(len(labels)>0):
        train_mem+=getsizeof(labels)
    train_mem/=gb
    
    eval_mem = getsizeof(eval_data)
    if(len(eval_labels)>0):
        eval_mem+=getsizeof(eval_labels)
    eval_mem/=gb
    
    print('Memory Consumed (process, train, eval) in GB : ',(process.memory_info().rss/gb), train_mem , eval_mem)  # in bytes 
