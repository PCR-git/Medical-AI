import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


def f_reshape_training_data(data):
    train0 = data[0,:,:].unsqueeze(0)
    train1 = data[1,:,:].unsqueeze(0)
    train2 = data[2,:,:].unsqueeze(0)
    train3 = data[3,:,:].unsqueeze(0)

    y1 = torch.concatenate((train0,train1),axis=0).unsqueeze(0)
    y2 = torch.concatenate((train1,train2),axis=0).unsqueeze(0)
    y3 = torch.concatenate((train2,train3),axis=0).unsqueeze(0)

    z1 = torch.concatenate((y1,y2,y3),axis=0).unsqueeze(0)
    
    return z1


def f_rotate(img, angle):
    
    sz = img.size()[0]

    # Convert to PIL, rotate, then convert back to tensor
    img_pil = TF.to_pil_image(img)
    img_pil_rot = TF.rotate(img_pil, angle)
    img_rot = TF.pil_to_tensor(img_pil_rot).squeeze().to(torch.float)

    if torch.max(img_rot) > 0:
        img_rot = (img_rot - torch.min(img_rot))/torch.max(img_rot)
        
    return img_rot

def f_rotate_and_zoom(img, angle):
    sz = img.size()[0]

    # Convert to PIL, rotate, then convert back to tensor
    img_pil = TF.to_pil_image(img)
    img_pil_rot = TF.rotate(img_pil, angle)
    img_rot = TF.pil_to_tensor(img_pil_rot).squeeze().to(torch.float)
#     img_rot = img_rot/torch.max(img_rot)
    
    theta = np.mod(angle,90)*(np.pi/180)
    scale = 1/np.cos(theta-np.pi/4)/np.sqrt(2)
    rm = scale + (1-scale)/2
    margin_rz = int(np.ceil(sz*rm))
    img_crop = img_rot[sz-margin_rz:margin_rz,sz-margin_rz:margin_rz]
    resize = transforms.Resize(size = sz,antialias=True)
    img_rsz = resize.forward(img_crop.unsqueeze(0)).squeeze()
    img_rsz = img_rsz * (torch.max(img)/ torch.max(img_rsz))
    
    return img_rsz


def f_random_crop(img_i,limits=[]):

    sz = img_i.size()[0]
    
    if len(limits) == 0:

        sum_y = (torch.sum(img_i,0)>0)*1
        y_l = torch.argmax(sum_y)
        sum_flip = torch.flip(sum_y,dims=(0,))
        ly_margin = torch.argmax(sum_flip)
        y_h = sz - ly_margin - 1
        y_sz = y_h - y_l

        sum_x = (torch.sum(img_i,1)>0)*1
        x_l = torch.argmax(sum_x)
        sum_flip = torch.flip(sum_x,dims=(0,))
        lx_margin = torch.argmax(sum_flip)
        x_h = sz - lx_margin - 1
        x_sz = x_h - x_l
        
#         img_z = img_i.clone()
#         img_z[x_l,y_l:y_h] = 1
#         img_z[x_h,y_l:y_h] = 1
#         img_z[x_l:x_h,y_l] = 1
#         img_z[x_l:x_h,y_h] = 1
#         plt.imshow(img_z)
#         plt.show()

        try:
            L_rand = random.randint(max(x_h-x_l, y_h-y_l), sz-1)
            x_rand = random.randint(max(0,x_h-L_rand-1), min(x_l-1,sz-1-L_rand))
            y_rand = random.randint(max(0,y_h-L_rand-1), min(y_l-1,sz-1-L_rand))
        except:
            print("Random crop failed")
            L_rand = sz-1
            x_rand = 0
            y_rand = 0
        limits = np.array([x_rand,y_rand,L_rand])

    else:
        
        x_rand = limits[0]
        y_rand = limits[1]
        L_rand = limits[2]

    img2 = img_i[x_rand:x_rand+L_rand,y_rand:y_rand+L_rand]

    resize = transforms.Resize(size = sz,antialias=True)
    img_rsz = resize.forward(img2.unsqueeze(0)).squeeze()
    img_rsz = img_rsz * (torch.max(img_i)/ torch.max(img_rsz))
    
    return img_rsz, limits


# i = 6
# img_v = train_data.data[i,:,:,:,:].squeeze().clone()

# img = img_v[0,0,:,256:512]
# plt.imshow(img)
# plt.show()

# for i in np.arange(10):
#     print(i)
# #     sum_y = (torch.sum(img,0)>0)*1
# #     sum_x = (torch.sum(img,1)>0)*1
# #     print(sum_x)
# #     print(sum_y)
    
#     img, lims = f_random_crop(img)
    
#     plt.imshow(img)
#     plt.show()


def f_rotate_and_zoom_all(img_v,dims,prob_rot=0.5):

    Nt = dims[1]
    J = dims[2]
    sz1 = dims[3]
    
    if np.random.uniform(0,1) < prob_rot:
        img_rot_v = torch.zeros_like(img_v)
        angle = random.randint(0,360)

        for jj in np.arange(J):
            for qq in np.arange(Nt):
                for kk in np.arange(3):
                    img_rot_v[qq,jj,:,sz1*kk:sz1*kk+sz1] = f_rotate_and_zoom(img_v[qq,jj,:,sz1*kk:sz1*kk+sz1].squeeze(), angle)
    else:
        img_rot_v = img_v
    
    return img_rot_v

def f_crop_all(img_v,dims,prob_crop=0.5):
    
    Nt = dims[1]
    J = dims[2]
    sz1 = dims[3]

    if np.random.uniform(0,1) < prob_crop:

        img_cropped_v = torch.zeros_like(img_v)

        img_cal = img_v[2,1,:,sz1:sz1*2]+img_v[2,1,:,sz1:sz1*2]+img_v[2,1,:,sz1:sz1*2]
        _, lims = f_random_crop(img_cal)

        for jj in np.arange(J):
            for qq in np.arange(Nt):
                for kk in np.arange(3):
                    img_cropped_v[qq,jj,:,sz1*kk:sz1*(kk+1)], _ = f_random_crop(img_v[qq,jj,:,sz1*kk:sz1*(kk+1)].squeeze(), lims)
    else:
        img_cropped_v = img_v
        
    return img_cropped_v


def f_flip_all(img_v,vflip,dims,prob_flip=0.5):

    Nt = dims[1]
    J = dims[2]
    sz1 = dims[3]

    if np.random.uniform(0,1) < prob_flip:
    
        img_flipped_v = torch.zeros_like(img_v)

        for jj in np.arange(J):
            for qq in np.arange(Nt):
                img_flipped_v[qq,jj,:,:] = vflip.forward(img_v[qq,jj,:,:].squeeze())
    else:
        img_flipped_v = img_v

    return img_flipped_v


def f_color_jitter_all(img_v, br = 0.7, cr = 0.7, prob_jit = 0.5):
    CJ = torchvision.transforms.ColorJitter(brightness = br, contrast = cr)

    img_cj = torch.zeros_like(img_v)
    img_cj[:,:,:,:] = img_v[:,:,:,:]

    j = 0
    k = 0
    if np.random.uniform(0,1) < prob_jit:
        x = img_v[j,k,:,0:256].unsqueeze(0)
        img_cj[j,k,:,0:256] = CJ.forward(x).squeeze()

    for j in np.arange(1):
        if np.random.uniform(0,1) < prob_jit:
            k = 0
            x = img_v[j,k,:,0:256].unsqueeze(0)
            img_cj[j,k,:,0:256] = CJ.forward(x).squeeze()
            k = 1
            j -= 1
            x = img_v[j,k,:,0:256].unsqueeze(0)
            img_cj[j,k,:,0:256] = CJ.forward(x).squeeze()
    
    j = 2
    k = 1
    if np.random.uniform(0,1) < prob_jit:
        x = img_v[j,k,:,0:256].unsqueeze(0)
        img_cj[j,k,:,0:256] = CJ.forward(x).squeeze()
    
    return img_cj


def f_augment_dataset2(train_data, num_pass = 5, prob_flip = 0.5, prob_rot = 0.25, prob_crop = 0.25, prob_jit = 0.5):

    dims = train_data.size()
    vflip = torchvision.transforms.RandomVerticalFlip(1)

    num_img = dims[0]
    train_data_augment = torch.zeros(num_img*num_pass,dims[1],dims[2],dims[3],dims[4])

#     for i_p in tqdm(np.arange(num_pass)):
    for i_p in np.arange(num_pass):
#         print("Pass:", i_p)
        for i_m in np.arange(num_img):

            i_t = i_m + num_img*i_p

            img_v = train_data[i_m,:,:,:,:].squeeze()

    #         plt.imshow(img_v[0,1,:,:])
    #         plt.show()

#             if i_p > 0:

            img_v = f_flip_all(img_v,vflip,dims,prob_flip)

#             plt.imshow(img_v[0,1,:,:])
#             plt.show()

            img_v = f_rotate_and_zoom_all(img_v,dims,prob_rot)

#             plt.imshow(img_v[0,1,:,:])
#             plt.show()

            img_v = f_crop_all(img_v,dims,prob_crop)
    
            
            img_v = f_color_jitter_all(img_v, br = 0.6, cr = 0.6, prob_jit = prob_jit)

#             plt.imshow(img_v[0,1,:,:])
#             plt.show()

            train_data_augment[i_t,:,:,:,:] = img_v
        
    return train_data_augment