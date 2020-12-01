# Prepares the input time series data for vitis-ai quantizer 
import numpy as np

data = np.load('calib_data.npz')['data']
batch_size=128

def calib_input(iter):

    calib_data = data[iter*batch_size:(iter+1)*batch_size]

    return {'input_1': calib_data}
