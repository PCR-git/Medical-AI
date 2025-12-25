import numpy as np
import torch
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
# import random
# import matplotlib.pyplot as plt
# from tqdm import tqdm

def f_rescale_dataset(unscaled_data):

    ud_shape = np.shape(unscaled_data)
    unscaled_data_flatten = np.reshape(unscaled_data, newshape = [ud_shape[0],ud_shape[1],ud_shape[2]*ud_shape[3]])

    # means = np.mean(GA_flatten,axis=2)
    stds  = np.sqrt(np.var(unscaled_data_flatten,axis=2))
    # means = np.expand_dims(np.expand_dims(means,axis=2),axis=3)

    mins = np.min(unscaled_data_flatten,axis=2)
    mins = np.expand_dims(np.expand_dims(mins,axis=2),axis=3)
    stds = np.expand_dims(np.expand_dims(stds,axis=2),axis=3)

    rescaled_data = (unscaled_data - mins)/stds
    
    return rescaled_data

def f_train_test_split(data_tot,data_mask,train_num):
    train_data = torch.from_numpy(np.expand_dims(data_tot, 1)[0:train_num,:,:,:,:])
    test_data = torch.from_numpy(np.expand_dims(data_tot, 1)[train_num:,:,:,:,:])

    train_data_mask = data_mask[0:train_num,:,:,:]
    test_data_mask = data_mask[train_num:,:,:,:]
    
    return train_data, test_data, train_data_mask, test_data_mask


def f_get_mask(frame_x):
    
    frame_x_pos = frame_x - np.min(frame_x)
    mask = 1*(frame_x_pos > 0.25*np.max(frame_x_pos))
    
    return mask


def f_get_all_masks(train_data_mask,test_data_mask,train_num,test_num,num_time_steps):

    train_data_mask_np = np.zeros_like(train_data_mask)
    test_data_mask_np = np.zeros_like(test_data_mask)

    for i in np.arange(train_num):
        for j in np.arange(num_time_steps):
             train_data_mask_np[i,j,:,:] = f_get_mask(train_data_mask[i,j,:,:])

    for i in np.arange(test_num):
        for j in np.arange(num_time_steps):
             test_data_mask_np[i,j,:,:] = f_get_mask(test_data_mask[i,j,:,:])

    train_data_mask = torch.from_numpy(np.expand_dims(train_data_mask_np,1))         
    test_data_mask = torch.from_numpy(np.expand_dims(test_data_mask_np,1))
    
    return train_data_mask, test_data_mask


def f_Residuals(data):
    Residuals = data[:,:,1:4,:,:] - data[:,:,0:3,:,:]
    NN = np.shape(Residuals)
    Z = np.zeros((NN[0], NN[1], 1, NN[3], NN[4])).astype(np.float32)
    Residuals_ = torch.from_numpy(np.concatenate((Z,Residuals),2))
    return Residuals_

