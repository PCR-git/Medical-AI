# import numpy as np
# import torch
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
# import random
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# def f_reshape_training_data(data):
#     train0 = data[0,:,:].unsqueeze(0)
#     train1 = data[1,:,:].unsqueeze(0)
#     train2 = data[2,:,:].unsqueeze(0)
#     train3 = data[3,:,:].unsqueeze(0)

#     y1 = torch.concatenate((train0,train1),axis=0).unsqueeze(0)
#     y2 = torch.concatenate((train1,train2),axis=0).unsqueeze(0)
#     y3 = torch.concatenate((train2,train3),axis=0).unsqueeze(0)

#     z1 = torch.concatenate((y1,y2,y3),axis=0).unsqueeze(0)
    
#     return z1

# def f_rotate(img, angle):
#     sz = img.size()[0]
    
#     # Zoom out so that nothing is cut off by rotation:
#     rsc = int(np.ceil((2**0.5)*sz/2)*2)
#     marg = int((rsc-sz)/2)
#     img_z = torch.zeros(rsc,rsc)
#     img_z[marg:rsc-marg,marg:rsc-marg] = img

#     # Convert to PIL, rotate, then convert back to tensor
#     img_pil = TF.to_pil_image(img_z)
#     img_pil_rot = TF.rotate(img_pil, angle)
#     img_rot = TF.pil_to_tensor(img_pil_rot).squeeze().to(torch.float)
    
#     resize = transforms.Resize(size = sz)
#     img_rsz = resize.forward(img_rot.unsqueeze(0)).squeeze()
    
#     return img_rsz

# def f_translate(img, trans=None):
#     if trans == None:
#         sz = img.size()[0]
#         mean_x = (torch.mean(img,0)>0)*1
#         l_margin = torch.argmax(mean_x)
#         mean_flip = torch.flip(mean_x,dims=(0,))
#         r_margin = torch.argmax(mean_flip)

#         margin = torch.min(l_margin, r_margin)

#         trans = random.uniform(0,margin)

#     img_shift = transforms.functional.affine(img.unsqueeze(2), angle = 0.0, translate = (0,trans), scale = 1.0, shear = 0.0).squeeze()
    
#     return img_shift, trans

# def f_resize(img, params=None, rm=1):

#     sz = img.size()[0]
    
#     if params == None:

#         mean_x = (torch.mean(img,0)>0)*1
#         x_l = torch.argmax(mean_x)
#         mean_flip = torch.flip(mean_x,dims=(0,))
#         r_margin = torch.argmax(mean_flip)
#         x_h = sz - r_margin
#         x_sz = x_h - x_l

#         mean_y = (torch.mean(img,1)>0)*1
#         y_l = torch.argmax(mean_y)
#         mean_flip = torch.flip(mean_y,dims=(0,))
#         d_margin = torch.argmax(mean_flip)
#         y_h = sz - d_margin
#         y_sz = y_h - y_l

#         max_sz = max(x_sz,y_sz)
#     #     dec = random.randint(0,int(torch.floor((sz - max_sz)/2)))
#         dec = random.randint(0,rm*int(torch.floor((sz - max_sz))))
#         dec_x = random.randint(0,dec)
#         dec_y = random.randint(0,dec)
        
#         img2 = torch.zeros((max_sz+dec,max_sz+dec))
#         diff = y_sz - x_sz

#         if y_sz > x_sz:
#             f2 = int(torch.floor(diff/2))
#             a = dec_y
#             b = dec_y + y_sz
#             c = dec_x + f2
#             d = dec_x + x_sz + f2

#     #         img2[dec_y:y_h-y_l+dec_y,f2+dec_x:x_h-x_l+f2+dec_x] = img[y_l:y_h,x_l:x_h]
#         else:
#             f2 = int(torch.floor(-diff/2))
#             a = dec_y + f2
#             b = dec_y + y_sz + f2
#             c = dec_x
#             d = dec_x + x_sz
            
#         params = [a,b,c,d,y_l,y_h,x_l,x_h, max_sz+dec]

#     #         img2[f2:y_h-y_l+f2,0:x_h-x_l] = img[y_l:y_h,x_l:x_h]
#     else:
#         a   = params[0]
#         b   = params[1]
#         c   = params[2]
#         d   = params[3]
#         y_l = params[4]
#         y_h = params[5]
#         x_l = params[6]
#         x_h = params[7]
#         msd = params[8]
        
#         img2 = torch.zeros((msd,msd))
    
#     img2[a:b,c:d] = img[y_l:y_h,x_l:x_h]
    
#     resize = transforms.Resize(size = sz)
#     img_rsz = resize.forward(img2.unsqueeze(0)).squeeze()
    
#     return img_rsz, params

# def f_augment_img(img):

#     Nt = img.size()[0]
#     sz = img.size()[1]

#     for ii in np.arange(Nt):
#         img[ii,:,:] /= torch.max(img[ii,:,:])

#     if random.uniform(0,1) > 0.5:
#         img_rsz = torch.zeros(Nt,sz,sz)
#         _, params = f_resize(2*img[0,:,:] + img[1,:,:], rm=0)
#         for ii in np.arange(Nt):
#             img_rsz[ii,:,:], _ = f_resize(img[ii,:,:], params, rm=0)
        
#         img_rot = torch.zeros(Nt,sz,sz)
#         angle = random.randint(-180,180)
#         for ii in np.arange(Nt):
#             img_rot[ii,:,:] = f_rotate(img[ii,:,:],angle)
        
#         img = img_rot

#     if random.uniform(0,1) > 0.5:
#         img_shift = torch.zeros(Nt,sz,sz)
#         img_shift[Nt-1,:,:] , trans = f_translate(img[Nt-1,:,:])
#         for ii in np.arange(Nt-1):
#             img_shift[ii,:,:] , _ = f_translate(img[ii,:,:], trans=trans)

#         img = img_shift

#     if random.uniform(0,1) > 0.5:
#         img_rsz = torch.zeros(Nt,sz,sz)
#         img_sum = 0
#         for ii in np.arange(Nt):
#             img_sum += img[ii,:,:]
#         _, params = f_resize(img_sum)

#         for ii in np.arange(Nt):
#             img_rsz[ii,:,:], _ = f_resize(img[ii,:,:], params)

#         img = img_rsz
    
#     return img

# def f_augment_dataset(train_data, num_pass=5):
#     num_img = np.shape(train_data.data)[0]

#     augmented_data = torch.zeros(tuple(np.append(num_img*num_pass,np.shape(train_data.data)[1:])))

#     for ii in tqdm(np.arange((num_pass-1)*num_img)):
#     # for ii in np.arange(10):

#         i = np.mod(ii,num_img)

#         img = train_data.data[i,:,:,:,:].squeeze()

#         plt.imshow(img[0,:,:] + img[1,:,:]/1.5, cmap="gray")
#         plt.show()
        
#         plt.imshow(img[1,:,:] + img[2,:,:]/1.5, cmap="gray")
#         plt.show()
        
#         plt.imshow(img[2,:,:] + img[3,:,:]/1.5, cmap="gray")
#         plt.show()
        
#         print('-------------------------')
        
#         img = f_augment_img(img)

#         augmented_data[ii,:,:,:,:] = img.unsqueeze(0)

#         plt.imshow(img[0,:,:] + img[1,:,:]/1.5, cmap="gray")
#         plt.show()
        
#         plt.imshow(img[1,:,:] + img[2,:,:]/1.5, cmap="gray")
#         plt.show()
        
#         plt.imshow(img[2,:,:] + img[3,:,:]/1.5, cmap="gray")
#         plt.show()
        
#         print('-------------------------')
#         print('-------------------------')

#     augmented_data[num_img*(num_pass-1):,:,:,:,:] = train_data.data
    
#     return augmented_data