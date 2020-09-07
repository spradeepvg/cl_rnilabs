#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 22:42:51 2019

@author: Saipradeep VG, RnI Labs 
"""
import time
import argparse
import numpy as np
import csv
import pandas as pd
import Utils as ut
import xgboost as xgb
import pickle

parser = argparse.ArgumentParser(description='Cell Lineage XGBoost SC#1 Training')
model_fname='xgb_triplet_label_sc1.model'
code_map={}

# Mandatory parameters
parser.add_argument('-i','--cl_dir', default='../data/', type=str,
                    help='cell lineage home directory')
parser.add_argument('-enc','--encoding', default='barcode', type=str,
                    help='Feature Encoding (barcode/hamming/hybrid)')
parser.add_argument('-ft','--full_train', default=True, type=bool,
                    help='Yes : Full training dataset for building model or No : 80/20 random split of train/eval ')
parser.add_argument('-t','--train_file', default='sc1_train.csv', type=str,
                    help='Train data file')
parser.add_argument('-e','--eval_file', default='sc1_eval.csv', type=str,
                    help='Eval data file')


def getPredictions(dataset, dataset_type, model):
    predictions = model.predict(dataset)
    probs_full = model.predict_proba(dataset)
    
    print(dataset_type,'predictions',len(predictions))
    print(dataset_type, 'probs',len(probs_full))
    return predictions, probs_full

def main():
    
    global args, code_map
    args = parser.parse_args()
    data_dir = args.cl_dir+'/sc1/'
    train_csv_file = data_dir+args.train_file
    eval_csv_file = data_dir+args.eval_file
    full_train = args.full_train
    full_train_csv_file = data_dir+'train_full.debug'
    t0= time.time()
    
    # Barcode - nest=50, mdepth=50, lr=0.3, gamma=0.5
    # Hamming - 'n_estimators': 60, 'max_depth': 30, 'learning_rate': 0.55, 'gamma': 0.18571428571428572
    optimal_params={'barcode': {'n_est':50, 'm_depth':50, 'lr':0.3, 'gamma':0.5},
                 'hammingcode':{'n_est':60, 'm_depth':30, 'lr':0.55, 'gamma':0.1857},
                 'hybrid':{'n_est':60, 'm_depth':30, 'lr':0.55, 'gamma':0.1857},
                 }
    
    # Hamming code or Bar code Approach
    print(' Building code map ')
    code_map = ut.build_SC1_CodeMap(data_dir, 76)
    print(' Created triplet vector map of size ...', len(code_map))
    #print(code_map)
    
    dim=32
    
    if full_train:
        col_idx=[0,2,3,4,5] # dreamID, [barcode], tc1, tc2, tc3, class
        train_csv_file=full_train_csv_file
        train, train_labels, train_idx = ut.getDataLoader(data_dir, train_csv_file, code_map, cols=col_idx, dim=dim, encoding=args.encoding)
        print(' index length ', len(train_idx))
        print(' Took ', time.time()-t0, ' seconds - Samples in Full Training ', len(train))        
    else:
        col_idx=[1,3,4,5,6] #[idx], dreamID, [barcode], tc1, tc2, tc3, class
        train, train_labels, train_idx = ut.getDataLoader(data_dir, train_csv_file, code_map, cols=col_idx, dim=dim, encoding=args.encoding)
        eval_data, eval_labels, eval_idx = ut.getDataLoader(data_dir, eval_csv_file, code_map, cols=col_idx, dim=dim, encoding=args.encoding)
        #eval_data = np.array(eval_data).squeeze(1)
        eval_data = np.array(eval_data)
        eval_labels = np.array(eval_labels)
        print(' Took ', time.time()-t0, ' seconds - Samples in Training ', len(train), ' Samples in Eval ', len(eval_data))
        
    train = np.array(train)
    train_labels = np.array(train_labels)
    
    code_map=None #GC
    
    #print(train.shape, eval_data.shape)
    print(' Train data shape',train.shape)
    
    p0 = time.time()
    # Create the model with the fine-tuned parameters after gridsearch
    params = optimal_params[args.encoding]
    model = xgb.XGBClassifier(n_estimators=params['n_est'], 
                               random_state=ut.RSEED,
                               max_depth=params['m_depth'],
                               learning_rate=params['lr'],
                               objective='multi:softprob',
                               gamma=params['gamma'],
                               n_jobs=-1, verbosity=3)
    
    print(' Training data ....', model.get_xgb_params())
    # Fit on training data
    model.fit(train, train_labels)    
    print(' Took ', time.time()-p0, ' seconds to train the model on ', len(train), ' rows')
    
    # Save model
    ut.saveModel(model, data_dir+model_fname)
    
    # Training predictions (to demonstrate overfitting)
    train_rf_predictions, train_rf_probs_full = getPredictions(train, 'train', model)
    
    if not full_train:
        # Testing eval predictions (to determine performance)
        eval_rf_predictions, eval_rf_probs_full = getPredictions(eval_data, 'eval', model)
    
    # Calculate metrics
    train_results = {}
    train_results['recall'], train_results['precision'] = ut.getMetrics(train_labels, train_rf_predictions)
    
    if not full_train:
        results = {}
        results['recall'], results['precision'] = ut.getMetrics(eval_labels, eval_rf_predictions)
        #for metric in ['recall', 'precision', 'roc']:
        for metric in ['recall', 'precision']:
            print(f'{metric.capitalize()} Eval: {round(results[metric], 4)} Train: {round(train_results[metric], 4)}')
        
        # Build confusion matrix
        from sklearn.metrics import confusion_matrix
        if not full_train:
            cm = confusion_matrix(eval_labels, eval_rf_predictions)
            print(cm)
            
        #print('Parameter estimation... ')
        # ut.fine_tune_xgb(model, train, train_labels, eval_data, eval_labels)
    else:
        for metric in ['recall', 'precision']:
            print(f'{metric.capitalize()} Train: {round(train_results[metric], 4)}')

    #xgb.plot_importance(model)
    #xgb.plot_tree(model)

    #Write predictions to output file    
    out_csv = open(data_dir+'sc1_xgb_triplet_label_train.out',mode='w')
    if not full_train:
        fieldnames = ['index', 'dreamID', 'barcode', 'triplet_cell1', 'triplet_cell2', 'triplet_cell3', 'true_class', 'predicted_class', 'prob', 'prob1', 'prob2', 'prob3']
        train_df = pd.read_csv(eval_csv_file)
    else:
        fieldnames = ['dreamID', 'barcode', 'triplet_cell1', 'triplet_cell2', 'triplet_cell3', 'true_class', 'predicted_class', 'prob', 'prob1', 'prob2', 'prob3']

    print('Writing train predictions ')    
    writer = csv.writer(out_csv, delimiter=',')
    writer.writerow(fieldnames)
    train_df = pd.read_csv(train_csv_file)
    ut.write_output(writer, train_df, train_idx, train_rf_predictions, train_rf_probs_full, ignoreIndexCol=full_train)
    out_csv.close()
    
    if not full_train:
        print('Writing validation predictions ')
        out_csv = open(data_dir+'sc1_xgb_triplet_label_eval.out',mode='w')
        writer = csv.writer(out_csv, delimiter=',')
        writer.writerow(fieldnames)
        eval_df = pd.read_csv(eval_csv_file)
        ut.write_output(writer, eval_df, eval_idx, eval_rf_predictions, eval_rf_probs_full, ignoreIndexCol=full_train)
        out_csv.close()
    
    print(' Took ', time.time()-t0, ' seconds ')
    
if __name__ == '__main__':
    main()