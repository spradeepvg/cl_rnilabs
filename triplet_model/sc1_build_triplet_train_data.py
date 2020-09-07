#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:48:05 2019

@author: sai
"""

import numpy as np
from newick import loads, read
import genTriplets
import argparse
import time
import pandas as pd
import csv
import h5py
from Triplet import Triplet


parser = argparse.ArgumentParser(description='Cell Lineage SC#1 Data Builder')

# Mandatory parameters
parser.add_argument('-i','--cl_dir', default='../data/', type=str,
                    help='cell lineage home directory')

def parseNewickTree(nwktree):
    tree = loads(nwktree)
    return tree

def readNewickFile(nwkfile):
    tree = read(nwkfile)
    return tree


def main():
    
    global args
    args = parser.parse_args()
    data_dir = args.cl_dir+'sc1/'
    csv_file = data_dir+'train_setDREAM2019.txt'
    t0=time.time()
    
    # Read the csv file
    full_data_info = pd.read_csv(csv_file, sep='\t')
    debugfile_csv = open(data_dir+'train_full.debug',mode='w')
    fieldnames = ['dreamID', 'barcode', 'triplet_cell1', 'triplet_cell2', 'triplet_cell3', 'class']
    writer = csv.writer(debugfile_csv, delimiter=',')
    writer.writerow(fieldnames)
    
    #outfile = h5py.File(data_dir+'train_full.hdf5', 'w')
    triplet_code_vecs =[]
    triplet_classes=[]
    colony_ids=[]
    dim=32
    
    #Iterate through each colony
    for idx in full_data_info.index:
    #for idx in range(0,1):
        colony_id = full_data_info.dreamID[idx]
        #num_cells = full_data_info.nCells[idx]
        
        # Parse the tree for each colony
        trees = parseNewickTree(full_data_info.ground[idx])
        
        # Build the triplet permutations from the nodes
        triplet_perms = genTriplets.buildTriplets(trees[0], isBarcodeNode=True) #Root node is the first element in the list
        print(len(triplet_perms))
        
        # Build the labels/classes for each permutation
        for triplet_perm in triplet_perms:
            # Assign the Class to the element added at the end (Class ids : 1-3)
            class_num = triplet_perm.true_class
            triplet_classes.append(class_num)
            
            # Build the triplet_barcode as numpy vector (1x32)
            #print(triplet_perm.vec)
            #triplet_code_vec = genTriplets.encodeVector(triplet_perm.vec, dim)
            #triplet_code_vecs.append(triplet_code_vec)
            
            # Add colony id for each permutation
            colony_ids.append(colony_id)
            
            #print(triplet_perm, ele_codes, class_num)
            writer.writerow([colony_id, triplet_perm.vec, triplet_perm.nodes[0], triplet_perm.nodes[1], triplet_perm.nodes[2], class_num])
    
    # Build the numpy vectors
    #triplet_code_arr = np.concatenate(triplet_code_vecs)
    #class_num_arr = np.array(triplet_classes)
    #colony_id_arr = np.array(colony_ids)
    
    # print('Colony ids shape=',colony_id_arr.shape)
    # print('Triplet code shape=',triplet_code_arr.shape)
    # print('Triplet class shape=',class_num_arr.shape)
    
    # Write the triplets and classes to output file
    #outfile.create_dataset('colony_id', data=colony_id_arr)
    #outfile.create_dataset('triplet_code', data=triplet_code_arr)
    #outfile.create_dataset('triplet_class', data=class_num_arr)
    #outfile.close()
    
    print(' Data creation took ',(time.time()-t0), ' seconds ')
    
if __name__ == '__main__':
    main()
