import pandas as pd
import h5py

filename = './data/train_data_loc.h5'

dset = h5py.File(filename, 'r')
print(dset["data"]["table"][0:10])
