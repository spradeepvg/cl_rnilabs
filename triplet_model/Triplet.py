#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:25:23 2020

@author: sai
"""

class Triplet():
    nodes=[]
    vec=''
    score=0.0
    true_class=0
    pred_class=0
    
    def __init__(self, nodes, true_class, vec='', pred_class=0, score=0):
        self.nodes=nodes
        self.vec=vec
        self.true_class=true_class
        self.pred_class=pred_class
        self.score=score
    
    