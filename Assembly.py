"""
Created 2nd November 2021

Author: Charlie Gerike-Roberts

Description: The assembly function is used to define how the nodes in the disassembled thermal circuits
are merged together.

Inputs: TCd

Outputs: AssX
"""
import pandas as pd
import numpy as np

# TCd=pd.DataFrame({'0': {'A': ([[-1,  0,  0,  0,  1],
#        [-1,  0,  0,  0,  1],
#        [-1,  0,  0,  0,  1],
#        [-1,  0,  0,  0,  1]]), 'G': ([[36.,  0.,  0.,  0.],
#        [ 0., 36.,  0.,  0.],
#        [ 0.,  0., 36.,  0.],
#        [ 0.,  0.,  0., 36.]]), 'b': ([0., 0., 0., 0.]), 'C': ([[    0.,     0.,     0.,     0.,     0.],
#        [    0.,     0.,     0.,     0.,     0.],
#        [    0.,     0.,     0.,     0.,     0.],
#        [    0.,     0.,     0.,     0.,     0.],
#        [    0.,     0.,     0.,     0., 16200.]]), 'f': ([0, 0, 0, 0, 1]), 'y': ([0, 0, 0, 0, 1])}, '1': {'A': ([[1],
#        [1]]), 'G': ([[  9.,   0.],
#        [  0., 500.]]), 'b': ([1, 1]), 'C': ([16200.]), 'f': 1, 'y': 1}, '2': {'A': ([[ 1., -0., -0., -0., -0.],
#        [-1.,  1., -0., -0., -0.],
#        [-0., -1.,  1., -0., -0.],
#        [-0., -0., -1.,  1., -0.],
#        [-0., -0., -0., -1.,  1.]]), 'G': ([[450.   ,   0.   ,   0.   ,   0.   ,   0.   ],
#        [  0.   , 630.   ,   0.   ,   0.   ,   0.   ],
#        [  0.   ,   0.   , 630.   ,   0.   ,   0.   ],
#        [  0.   ,   0.   ,   0.   ,  30.375,   0.   ],
#        [  0.   ,   0.   ,   0.   ,   0.   ,  30.375]]), 'b': ([1., 0., 0., 0., 0.]), 'C': ([[       0.,        0.,        0.,        0.,        0.],
#        [       0., 18216000.,        0.,        0.,        0.],
#        [       0.,        0.,        0.,        0.,        0.],
#        [       0.,        0.,        0.,   239580.,        0.],
#        [       0.,        0.,        0.,        0.,        0.]]), 'f': ([1., 0., 0., 0., 1.]), 'y': ([0., 0., 0., 0., 0.])}, '3': {'A': ([[ 1,  0],
#        [-1,  1]]), 'G': ([[ 78.75,   0.  ],
#        [  0.  , 630.  ]]), 'b': ([1, 0]), 'C': ([[675000.,      0.],
#        [     0.,      0.]]), 'f': ([1, 0]), 'y': ([0, 0])}, '4': {'A': ([[-1,  0,  0],
#        [-1,  1,  0],
#        [ 0, -1,  1]]), 'G': ([[450.   ,   0.   ,   0.   ],
#        [  0.   ,  30.375,   0.   ],
#        [  0.   ,   0.   ,  30.375]]), 'b': ([1, 0, 0]), 'C': ([[     0.,      0.,      0.],
#        [     0., 239580.,      0.],
#        [     0.,      0.,      0.]]), 'f': ([1, 0, 1]), 'y': ([0, 0, 0])}, '5': {'A': ([[ 1,  0,  0,  0],
#        [-1,  1,  0,  0],
#        [ 0, -1,  1,  0],
#        [ 0,  0, -1,  1]]), 'G': ([[180.,   0.,   0.,   0.],
#        [  0., 180.,   0.,   0.],
#        [  0.,   0., 315.,   0.],
#        [  0.,   0.,   0., 315.]]), 'b': ([1, 0, 0, 0]), 'C': ([[   32400.,        0.,        0.,        0.],
#        [       0.,        0.,        0.,        0.],
#        [       0.,        0., 18216000.,        0.],
#        [       0.,        0.,        0.,        0.]]), 'f': ([0, 0, 0, 1]), 'y': ([0, 0, 0, 0])}})

def assembly(TCd):

       TCd_last_node = np.zeros(len(TCd)-1)  # define size of matrix for last node in each TC
       TCd_element_numbers = np.arange(1, len(TCd), 1)  #create vector which contains the number for each element

       # compute number of last node of each thermal circuit and input into thermal circuit sizes matrix
       for i in range(0, len([TCd_last_node][0])):
           TCd_last_node[i] = len(TCd[str(i+1)]['A'][0])-1

       print(TCd_last_node)

       IA_nodes = np.arange(len(TCd[str(0)]['A'][0]))  #create vector with the nodes for inside air
       print(IA_nodes)

       # create assembly matrix
       AssX = np.zeros((len(IA_nodes), 4))  #define size of AssX matrix
       for i in range(0, len([AssX][0])):
           AssX[i, 0] = TCd_element_numbers[i]  #set first column of row to element
           AssX[i, 1] = TCd_last_node[i]   #set second column to last node of that element
           AssX[i, 2] = 0     #set third column to inside air element
           AssX[i, 3] = IA_nodes[i]    #set 4th column to element of inside air which connects to corresponding element

       AssX = AssX.astype(int)

       print(AssX)

       return AssX
