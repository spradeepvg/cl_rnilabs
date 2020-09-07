#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:48:05 2019

@author: sai
"""

import numpy as np
from itertools import permutations
import argparse
import time
import os
import fnmatch
import csv
import h5py
import Utils as ut

parser = argparse.ArgumentParser(description='Cell Lineage SC#1 Test Data Builder')

# Mandatory parameters
parser.add_argument('-i','--cl_dir', default='../data/', type=str,
                    help='cell lineage home directory')

def getFiles(test_dir):
    test_files = []
    for count, f_name in enumerate(sorted(os.listdir('/home/sai/challenges/cell_lineage/data/sc1/sc1_test_txt/')), start=1):
        if fnmatch.fnmatch(f_name, '*_test_*.txt'):
            test_files.append(f_name)
    return test_files

def getPermutations(nodeids, r=3):
    triplet_perms=[]
    cell_perm = permutations(nodeids, r)
    for i in list(cell_perm):
        triplet_perms.append(i)
    return triplet_perms

def getColonyID(fname, prefix, suffix):
    return int(fname[fname.index(prefix)+len(prefix):fname.index(suffix)])

def main():
    
    global args
    args = parser.parse_args()
    data_dir = args.cl_dir+'sc1/sc1_test_txt/'
    t0=time.time()
    
    # Read the csv file
    debugfile_csv = open(data_dir+'test_full.debug',mode='w')
    fieldnames = ['dreamID', 'barcode', 'triplet_cell1', 'triplet_cell2', 'triplet_cell3']
    writer = csv.writer(debugfile_csv, delimiter=',')
    writer.writerow(fieldnames)
    
    outfile = h5py.File(data_dir+'test_full.hdf5', 'w')
    triplet_code_vecs =[]
    colony_ids=[]
    
    test_files = getFiles(data_dir)
    print(test_files)
    
    #Iterate through each colony test file
    for test_file in test_files:
        print(' Building test data for ', test_file)
        colony_id = getColonyID(test_file, 'test_', '.txt')
        df, num_cells = ut.CLTextFileReader(data_dir, test_file)
        nseq = df.cell.values
        
        # Get the triplet permutations for the nodes
        triplet_perms = getPermutations(nseq)
        print(colony_id, num_cells, nseq, len(triplet_perms))
        
        # Build the labels/classes for each permutation
        for triplet_perm in triplet_perms:
            triplet_code=''
            for ele in triplet_perm:
                curr_code = df[df.cell==ele].state.values[0]
                if(len(triplet_code)>0):
                    triplet_code = triplet_code + '_' + curr_code
                else:
                    triplet_code = curr_code
            
            #print(triplet_perm, triplet_code)
            # Build the triplet_barcode as numpy vector (1x32)
            triplet_code_b = triplet_code.encode()
            triplet_code_vec = np.frombuffer(triplet_code_b, dtype=np.uint8).reshape(1,32)
            triplet_code_vecs.append(triplet_code_vec)
            
            # Add colony id for each permutation
            colony_ids.append(colony_id)
            
            #print('\t',triplet_perm, triplet_code_vec)
            writer.writerow([colony_id, triplet_code, triplet_perm[0], triplet_perm[1], triplet_perm[2]])
    
    # Build the numpy vectors
    triplet_code_arr = np.concatenate(triplet_code_vecs)
    colony_id_arr = np.array(colony_ids)
    
    print('Colony ids shape=',colony_id_arr.shape)
    print('Triplet code shape=',triplet_code_arr.shape)
    
    # Write the triplets and classes to output file
    outfile.create_dataset('colony_id', data=colony_id_arr)
    outfile.create_dataset('triplet_code', data=triplet_code_arr)
    outfile.close()
    
    print(' Data creation took ',(time.time()-t0), ' seconds ')
    
if __name__ == '__main__':
    main()
