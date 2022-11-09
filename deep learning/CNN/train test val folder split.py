# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 20:30:56 2022

@author: SSROY
"""

import os
import splitfolders
# train validation test split
input_folder = r"D:\image classification\done\random model"

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#Train, val, test
splitfolders.ratio(input_folder, output=r"D:\image classification\done\random model", 
                   seed=42, ratio=(.7, .2,.1),                   # (.7, .2, .1)
                   group_prefix=None) # default values