#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 22:42:51 2019

@author: sai
"""
import time
import argparse
import h5py
import numpy as np
import csv
import pandas as pd
import Utils as ut
import xgboost as xgb
import pickle

parser = argparse.ArgumentParser(description='Cell Lineage XGBoost SC#1 Training')


# Mandatory parameters
parser.add_argument('-i','--cl_dir', default='../data/', type=str,
                    help='cell lineage home directory')
parser.add_argument('-m','--model', default='sc1_xgb_triplet_label.model', type=str,
                    help='cell lineage home directory')
parser.add_argument('-o','--output', default='sc1_xgb_triplet_label_test.out', type=str,
                    help='cell lineage home directory')
parser.add_argument('-enc','--encoding', default='hammingcode', type=str,
                    help='Feature Encoding (barcode/hamming/hybrid)')
    
def getModel(mfname):
    return pickle.load(open(mfname, "rb"))

def getPredictions(dataset, dataset_type, model):
    predictions = model.predict(dataset)
    probs_full = model.predict_proba(dataset)
    
    print(dataset_type,'predictions',len(predictions))
    print(dataset_type, 'probs',len(probs_full))
    return predictions, probs_full

def main():
    
    global args
    args = parser.parse_args()
    data_dir = args.cl_dir+'sc1/'
    model_fname=args.model
    output_fname=args.output
    inp_file = data_dir+'test_full.hdf5'
    debug_csv_file = data_dir+'test_full.debug'
    
    t0= time.time()
    # Read the debug file
    full_data_info = pd.read_csv(debug_csv_file)
    
    code_map = ut.build_SC1_TestCodeMap(data_dir, 30)
    col_idx=[0,2,3,4] # dreamID, [barcode], tc1, tc2, tc3, class
    test_csv_file=debug_csv_file
    test, test_labels, test_idx = ut.getDataLoader(data_dir, test_csv_file, code_map, cols=col_idx, dim=32, encoding=args.encoding)
    print(' index length ', len(test_idx))    
    indices = np.arange(len(test))
    
    p0 = time.time()
    # Create the model with the fine-tuned parameters after gridsearch
    model = getModel(data_dir+model_fname)
    
    # Training predictions (to demonstrate overfitting)
    predictions, probs_full = getPredictions(test, 'test', model)
    
    # Write predictions to output file
    out_csv = open(data_dir+output_fname, mode='w')
    writer = csv.writer(out_csv, delimiter=',')
    fieldnames = ['dreamID', 'barcode', 'triplet_cell1', 'triplet_cell2', 'triplet_cell3', 'predicted_class', 'prob', 'prob1', 'prob2', 'prob3']
    writer.writerow(fieldnames)
    
    print('Writing test predictions ')
    ut.write_test_output(writer, full_data_info, indices, predictions, probs_full)
    
    print(' Took ', time.time()-t0, ' seconds ')
    
if __name__ == '__main__':
    main()
