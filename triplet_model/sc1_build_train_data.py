#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:09:33 2020

@author: sai
"""


import numpy as np
import argparse
import time
import pandas as pd
import csv
import dendropy
import Utils

parser = argparse.ArgumentParser(description='Cell Lineage SC#2 Data Builder')

# Mandatory parameters
parser.add_argument('-i','--cl_dir', default='/home/sai/cell_lineage', type=str,
                    help='cell lineage home directory')

def buildDictFromDF(df, kcol, vcol):
    dict={}
    for idx in range(0, len(df)):
        dict[df[kcol].iloc[idx]]=df[vcol].iloc[idx]
    return dict


def main():
    
    global args
    args = parser.parse_args()
    data_dir = args.cl_dir+'/data/sc1/'
    t0=time.time()
    
    num_colonies = 106
    targed_enc_units = 10
    
    debugfile_csv = open(data_dir+'sc1_train_full.debug',mode='w')
    fieldnames = ['dreamID', 'code1', 'code2', 'cell1', 'cell2', 'distance']
    writer = csv.writer(debugfile_csv, delimiter=',')
    writer.writerow(fieldnames)    
    
    #Iterate through each of the 100 colonies
    for idx in range(1,num_colonies+1):
        cid = str(idx)
        csv_file = data_dir+'sc1_train_txt/sub1_train_'+cid+'.txt'
        # Read the cell-barcode map csv file
        cell_barcode_df = pd.read_csv(csv_file, sep='\t', header=None)
        #cell_barcode_map = cell_barcode_df.to_dict()
        cell_barcode_map = buildDictFromDF(cell_barcode_df, 0, 1)
        
        #print(cell_barcode_map)
        
        # Read the ground truth from the newick file
        newick_ref_file = data_dir+'ref_train/sub1_train_'+cid+'.nwk'
        
        # Parse the tree for each colony
        tree = dendropy.Tree.get(path=newick_ref_file, schema="newick")
        pdc = tree.phylogenetic_distance_matrix()
        
        # Build the triplet permutations from the nodes
        triplet_perms = genTriplets.buildTriplets(tree[0]) #Root node is the first element in the list
        
        # Build the labels/classes for each permutation
        i=0
        for triplet_perm in triplet_perms:
            # Assign the Class to the element added at the end (Class ids : 1-3)
            class_num = triplet_perm.true_class
            #tlset[idx-1,i]=class_num
            
            # Build the triplet_barcode as numpy vector (1x32)
            #triplet_vec = genTriplets.getTripletVector(triplet_perm.nodes,cell_barcode_map, dim)
            #tvec = np.string_(triplet_vec)
            #tcset[idx-1,i]=tvec
            
            # Add colony id for each permutation
            #cidset[idx-1,i]=np.string_(cid)
            
            #print(triplet_perm, ele_codes, class_num)
            writer.writerow([idx, triplet_perm.nodes[0], triplet_perm.nodes[1], triplet_perm.nodes[2], class_num])
            
            i+=1
            
        print('Took ',(time.time()-t0), ' seconds to write ', i, ' triplet permutations for {', cid, len(triplet_perms),'}')
            
    #print('Colony ids shape=',colony_id_arr.shape)
    #print('Triplet code shape=',triplet_code_arr.shape)
    #print('Triplet class shape=',class_num_arr.shape)
    
    #outfile.close()
    
    print(' Data creation took ',(time.time()-t0), ' seconds ')
    
if __name__ == '__main__':
    main()
