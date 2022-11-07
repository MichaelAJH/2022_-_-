import pandas as pd
import h5py

filename = './data/data.h5'

dset = h5py.File(filename, 'r')
print(dset["data"]["table"][2])
