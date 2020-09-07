#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 18:30:49 2020

@author: sai
"""


class CNode():
    left=None
    right=None
    id=0
    code=''
    is_leaf=False
    wt=0.0
    
    def __init__(self, left, right, id=0, code='', weight=0, leaf=False):
        self.left=left
        self.right=right
        self.id=id
        self.code=code
        self.wt=weight
        self.is_leaf = leaf
        
    def setLeaf(self, leaf=True):
        self.is_leaf=leaf