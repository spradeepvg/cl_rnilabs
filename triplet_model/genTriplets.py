#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:06:50 2020

@author: sai
"""

from newick import loads, read
import newick
from Triplet import Triplet
from CNode import CNode
import numpy as np

# UNMUTATED=0
# MATCH_MUTATED=1
# UNMATCH_MUTATED=2
# DELIM=3

# ASCII Values of 0,1,2 and '_'  
UNMUTATED=48
MATCH_MUTATED=49
UNMATCH_MUTATED=50
DELIM=95

# UNMUTATED=b'0'
# MATCH_MUTATED=b'1'
# UNMATCH_MUTATED=b'2'
# DELIM=b'3'

def encodeVector(triplet_code, dim):
    triplet_code_b = triplet_code.encode()
    triplet_code_vec = np.frombuffer(triplet_code_b, dtype=np.uint8).reshape(1,dim)
    return triplet_code_vec

def concateVector(v1, v2, v3):
    triplet_code=v1+'_'+v2+'_'+v3
    return triplet_code

def hammingVector(v1, v2, v3, triplet_perm):
    assert (len(v1)==len(v2)==len(v3))
    
    h12=[]
    h23=[]
    h31=[]
    
    for i in range(len(v1)):
        h12.append(compareVecPair(v1[i],v2[i]))
        h23.append(compareVecPair(v2[i],v3[i]))
        h31.append(compareVecPair(v3[i],v1[i]))
    
    triplet_code = np.concatenate((np.array(h12, dtype=np.int8()),
                                   np.array([DELIM], dtype=np.int8()),
                                  np.array(h23, dtype=np.int8()),
                                  np.array([DELIM], dtype=np.int8()),
                                  np.array(h31, dtype=np.int8())))
    #print(triplet_code.shape)
    return triplet_code


def compareVecPair(v1, v2):
    if(v1==v2):
        if(v1=='0'):
            return UNMUTATED
        else:
            return MATCH_MUTATED
    else:
        return UNMATCH_MUTATED

def getTripletVector(triplet_id, cb_map, dim):
    triplet_code = concateVector(cb_map[triplet_id[0]], cb_map[triplet_id[1]], cb_map[triplet_id[2]])
    # print(len(triplet_code), triplet_code)
    return triplet_code

def getNodeInfo(node):
    node_name = node.name
    idx = node_name.index('_')
    return node_name[0:idx], node_name[idx+1:len(node_name)]

def getNodeInfofromDF(node):
    node_id = node.name
    return node_id

def traverse(node, cnode, isBarcodeNode=False):
    if(node==None):
        return cnode
    
    if node.is_leaf==True:
        if(cnode!=None):
            if(isBarcodeNode):
                node_id, node_code = getNodeInfo(node)
                cnode = CNode(None, None, id=node_id, code=node_code, leaf=True)
            else:    
                node_id = getNodeInfofromDF(node)
                #print(node_id)
                cnode = CNode(None, None, id=node_id, leaf=True)            
        return cnode
    
    #print('Processing node ', node)
    d_nodes = node.descendants
    cnode = CNode(None, None)
    
    #print(d_nodes)
    lnode = traverse(d_nodes[0], cnode, isBarcodeNode=isBarcodeNode)
    rnode = traverse(d_nodes[1], cnode, isBarcodeNode=isBarcodeNode)
    cnode.left = lnode
    cnode.right = rnode
    
    return cnode
    
def walk(cnode):
    if(cnode==None):
        return
    
    if cnode.is_leaf==True:
        print('\t',cnode.id, cnode.code)
    
    if(cnode.left!=None):
        print(' Traversing left nodes ', cnode.id)
        walk(cnode.left)
    if(cnode.right!=None):
        print(' Traversing right nodes ', cnode.id)
        walk(cnode.right)
    return


def genTriplets(node, triplets, nodes, isBarcodeNode=False):
    if(node==None):
        return
    #if node.is_leaf==True:
    if (node.left==None or node.right==None):
        #print('\tLeaf node ', node.id, ' code =',node.code)
        nodes.append(node)
        return
    
    #print('Current Node ', node.id)
    
    lnodes=[]
    rnodes=[]
    
    genTriplets(node.left, triplets, lnodes, isBarcodeNode=isBarcodeNode) #Get Left sub-tree of the Root
    genTriplets(node.right, triplets, rnodes, isBarcodeNode=isBarcodeNode) #Get Right sub-tree of the Root
    
    #print('lnodes =',len(lnodes),'rnodes =',len(rnodes))
    
    for ln_info1 in lnodes:
        cell_id1=ln_info1.id
        cell_v1=ln_info1.code
        for ln_info2 in lnodes:
            cell_id2=ln_info2.id
            if(cell_id1==cell_id2):
                continue
            cell_v2=ln_info2.code
            for rn_info in rnodes:
                cell_id3=rn_info.id
                cell_v3=rn_info.code
                #print(cell_id1, cell_id2, cell_id3)
                
                if(isBarcodeNode):
                    triplets.append(Triplet([cell_id1, cell_id2, cell_id3], 3, vec=concateVector(cell_v1, cell_v2, cell_v3)))
                    triplets.append(Triplet([cell_id1, cell_id3, cell_id2], 2, vec=concateVector(cell_v1, cell_v3, cell_v2)))
                    triplets.append(Triplet([cell_id3, cell_id1, cell_id2], 1, vec=concateVector(cell_v3, cell_v1, cell_v2)))
                else:
                    triplets.append(Triplet([cell_id1, cell_id2, cell_id3], 3))
                    triplets.append(Triplet([cell_id1, cell_id3, cell_id2], 2))
                    triplets.append(Triplet([cell_id3, cell_id1, cell_id2], 1))
                
    
    
    #print()
    
    for rn_info1 in rnodes:
        cell_id1=rn_info1.id
        cell_v1=rn_info1.code
        for rn_info2 in rnodes:
            cell_id2=rn_info2.id
            if(cell_id1==cell_id2):
                continue
            cell_v2=rn_info2.code
            for ln_info in lnodes:
                cell_id3=ln_info.id
                cell_v3=ln_info.code
                #print(cell_id1, cell_id2, cell_id3)
                if(isBarcodeNode):
                    triplets.append(Triplet([cell_id1, cell_id2, cell_id3], 3, vec=concateVector(cell_v1, cell_v2, cell_v3)))
                    triplets.append(Triplet([cell_id1, cell_id3, cell_id2], 2, vec=concateVector(cell_v1, cell_v3, cell_v2)))
                    triplets.append(Triplet([cell_id3, cell_id1, cell_id2], 1, vec=concateVector(cell_v3, cell_v1, cell_v2)))
                else:
                    triplets.append(Triplet([cell_id1, cell_id2, cell_id3], 3))
                    triplets.append(Triplet([cell_id1, cell_id3, cell_id2], 2))
                    triplets.append(Triplet([cell_id3, cell_id1, cell_id2], 1))
    
    nodes.extend(lnodes)
    nodes.extend(rnodes)
    #print(triplets)
    
    return


def buildTriplets(newick_tree, isBarcodeNode=False):
    
    """
    
    Parameters
    ----------
    newick_tree : Tree (Newick)
        Takes as input the Newick Parsed Tree and return breadth-first traversal of the tree .

    Returns
    -------
    List of Node-names in breadth-first sequence.

    """
    
    cnode = CNode(None, None)
    print(' Converting Tree ...')
    cnode=traverse(newick_tree, cnode, isBarcodeNode=isBarcodeNode)
    #walk(cnode)
    #print(' Done ....')
    
    triplets=[]
    nodes=[]
    print(' Building Triplets ....')
    genTriplets(cnode, triplets, nodes, isBarcodeNode=isBarcodeNode)
    #print(len(triplets))
    
    return triplets
    
    
if __name__ == '__main__':
    trees = loads('((((1_2012212021:8,2_2112212021:8):38,(3_2112212021:4,4_2112212021:4):42):42,(5_0012212221:1,6_0012012221:1):87):46,((8_2120010021:22,9_2120010021:22):51,10_0112212221:74):62);')
    buildTriplets(trees[0])
