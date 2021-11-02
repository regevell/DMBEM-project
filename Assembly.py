"""
Created 2nd November 2021

Author: Charlie Gerike-Roberts

Description: The assembly function is used to define how the nodes in the disassembled thermal circuits
are merged together.

Inputs: TCd

Outputs: AssX
"""

def Assembly(TCd):

    TCd_size = TCd.f.apply(lambda x: np.size(x)) # define size of thermal circuit sizes matrix
    # define number of nodes in inside air thermal circuit.

    # compute size of each thermal circuit and input into thermal circuit sizes matrix
    for i in 1 : size(TCd_size)
        TCd_size(i) =


    return AssX