import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from PIL import Image
from tabulate import tabulate

def f_get_predictions(VPTR_Transformer,VPTR_Enc,VPTR_Dec,sample,device):
    VPTR_Transformer_E = VPTR_Transformer.eval()
    with torch.no_grad():

#         past_frames = sample[:,:,0,:,:].unsqueeze(2).to(device)
#         future_frames = sample[:,:,1,:,:].unsqueeze(2).to(device)

        sample = torch.moveaxis(sample, 2, 0)
        past_frames, future_frames = sample
        past_frames = past_frames.unsqueeze(2).to(device)
        future_frames = future_frames.unsqueeze(2).to(device)

        past_gt_feats = VPTR_Enc(past_frames)
#         future_gt_feats = VPTR_Enc(future_frames)

#         rec_past_frames = VPTR_Dec(past_gt_feats)
#         rec_future_frames = VPTR_Dec(future_gt_feats)

        pred_future_feats = VPTR_Transformer_E(past_gt_feats)
        pred_future_frames = VPTR_Dec(pred_future_feats)

#         future_frames = future_frames[:,:,:,:,256:]
#         pred_future_frames = pred_future_frames[:,:,:,:,256:]
        
        return past_frames, future_frames, pred_future_frames
    
def f_plot_side_by_side(future_frames, pred_future_frames,samp=0):

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for idx, ax in enumerate(axes.flat):
        ax.imshow(future_frames[samp,idx,0,:,:].detach().cpu().numpy())
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for idx, ax in enumerate(axes.flat):
        ax.imshow(pred_future_frames[samp,idx,0,:,:].detach().cpu().numpy())
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")
        
def f_plot_overlay(future_frames,pred_future_frames,samp,idx,ax,alpha=0.5):
    x = future_frames[samp,idx,0,:,:].detach().cpu().numpy()
    y = pred_future_frames[samp,idx,0,:,:].detach().cpu().numpy()

    x = x - np.min(x)
    y = y - np.min(y)
    x = x/np.max(x)
    y = y/np.max(y)
    # z = x*y
    z = x*y
    z = z - np.min(z)
    z = z/np.max(z)
    # z = z>np.max(z)*0.9
    ax.imshow(x, cmap="gray", alpha = alpha)
    ax.imshow(y-z, cmap="gray", alpha = alpha)
    
    ax.set_title(f"Frame {idx + 1}")
    ax.axis("off")
    
#     plt.show()

# Plot symmetric difference
# https://en.wikipedia.org/wiki/Symmetric_difference
def f_plot_symmetric_difference(frame_x,frame_y,idx,ax,alpha=0.5):

    frame_x = frame_x - np.min(frame_x)
    frame_y = frame_y - np.min(frame_y)
    frame_x = frame_x/np.max(frame_x)
    frame_y = frame_y/np.max(frame_y)
    
    frame_z = frame_x*frame_y
    frame_z = frame_z - np.min(frame_z)
    frame_z = frame_z/np.max(frame_z)
    
    sym_diff = frame_x + frame_y - 2*frame_z
    sym_diff = sym_diff > 0.5
    
    ax.imshow(sym_diff, cmap="gray", alpha = alpha)
    
    ax.set_title(f"Frame {idx + 1}")
    ax.axis("off")
    
#     plt.show()

def f_plot_difference(frame_x,frame_y,idx,ax,alpha=0.5):

    diff = frame_y - frame_x
    ax.imshow(diff, cmap="gray")
    ax.set_title(f"Frame {idx + 1}")
    ax.axis("off")
    
# fig, axes = plt.subplots(1, 2, figsize=(14, 4))
# frame_x = future_frames[samp,idx+1,0,:,:]
# frame_y = future_frames[samp,idx,0,:,:]

def f_plot_diff_same_t(future_frames,pred_future_frames,samp):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for idx, ax in enumerate(axes.flat):
        frame_x = future_frames[samp,idx,0,:,:].detach().cpu().numpy()
        frame_y = pred_future_frames[samp,idx,0,:,:].detach().cpu().numpy()
        
        frame_x -= np.min(frame_x)
        frame_x /= np.max(frame_x)
        frame_y -= np.min(frame_y)
        frame_y /= np.max(frame_y)

        f_plot_difference(frame_x,frame_y,idx,ax,alpha=0.7)
#         f_plot_symmetric_difference(frame_x,frame_y,idx,ax,alpha=0.7)

def f_save_diff(frame_x,frame_y,ii,pred_save_dir):
    
    diff = frame_y - frame_x
    
    diff -= np.min(diff)
    diff *= 255/np.max(diff)

    diff = diff.astype(np.uint8)
    
#     fig, ax = plt.subplots()
    plt.imshow(diff)
    plt.show()
#     fig.clear(); del fig

    data = Image.fromarray(diff)
    save_name = pred_save_dir + "img_pred_error" + "_" + str(int(ii)) + ".jpg"
    data.save(save_name)
    
    return diff

# def f_get_mask(frame_x):
    
#     frame_x = frame_x - np.min(frame_x)
#     mask = 1*(frame_x > 0.65*np.max(frame_x))
    
#     return mask

def f_get_mask(frame_x):
    
    frame_x_pos = frame_x*(frame_x>0)
#     frame_x_neg = frame_x*(frame_x<0)
    
    frame_x_pos = frame_x_pos - np.min(frame_x_pos)
    mask = 1*(frame_x_pos > 0.65*np.max(frame_x_pos))
    
#     frame_x_neg = frame_x_neg - np.max(frame_x_neg)
#     mask += 1*(frame_x_neg < 0.65*np.min(frame_x_neg))
    
    return mask

def f_overlap(frame_x,frame_y):
    
#     mask1 = f_get_mask(frame_x)
#     mask2 = f_get_mask(frame_y)
    
#     mask1 = frame_x > 0.5
#     mask2 = frame_y > 0.5

    mask1 = frame_x
    mask2 = frame_y
    
    union = 1*((mask1 + mask2) > 0)
    
    intersect = (mask1*mask2)

    nonzero_intersect = cv2.countNonZero(intersect)
    nonzero_union = cv2.countNonZero(union)
    nonzero_sum = cv2.countNonZero(mask1) + cv2.countNonZero(mask2)
    
#     intersect_area = nonzero_intersect/(intersect.shape[0]*intersect.shape[1])
#     union_area = nonzero_union/(union.shape[0]*union.shape[1])
    
    jaccard = nonzero_intersect/nonzero_union
    dice = 2*nonzero_intersect/nonzero_sum
    
    return intersect, union, jaccard, dice

def f_diff_subsequent_t(future_frames,pred_future_frames,samp,ii,pred_save_dir):
    
    jaccard = np.zeros(2)
    dice = np.zeros(2)
   
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for idx in np.arange(2):
        frame_x = future_frames[samp,idx,0,:,256:512].detach().cpu().numpy()
        frame_y = future_frames[samp,idx+1,0,:,256:512].detach().cpu().numpy()

        diff = frame_y - frame_x
        diff *= (diff > 0)
        
        pred_frame_x = pred_future_frames[samp,idx,0,:,256:512].detach().cpu().numpy()
        pred_frame_y = pred_future_frames[samp,idx+1,0,:,256:512].detach().cpu().numpy()

#         pred_diff = pred_frame_y - pred_frame_x
        pred_diff = pred_frame_y - frame_x
        pred_diff *= (pred_diff > 0)

        intersect, union, jac, dc = f_overlap(diff,pred_diff)
        jaccard[idx] = np.round(jac*100,2)
        dice[idx] = np.round(dc*100,2)
        
        out = np.array(intersect + union)
        out = out - np.min(out)
        out = 255*out/np.max(out)
        
#         axes[int(idx),0].imshow(diff)
#         axes[int(idx),0].set_title(f"Frame {idx + 1}")
        
#         axes[int(idx),1].imshow(pred_diff)
#         axes[int(idx),1].set_title(f"Frame {idx + 1}")
       
        axes[0,int(idx)].imshow(f_get_mask(diff))
        axes[0,int(idx)].set_title(f"Groundtruth Diff, Frame {idx + 1}")
        
        axes[1,int(idx)].imshow(f_get_mask(pred_diff))
        axes[1,int(idx)].set_title(f"Predicted Diff, Frame {idx + 1}")
        
#         fig, ax = plt.subplots()
#         ax.imshow(intersect + union, cmap="gray")
# #         plt.title("Intersection and Union")
#         ax.axis("off")
#         plt.show()
               
        data = Image.fromarray(out.astype(np.uint8))
        save_name = pred_save_dir + "Growth/intersect_union" + "_" + str(int(ii)) + "_" + str(idx) + ".jpg"
        data.save(save_name)
    
    _, _, baseline_jac, baseline_dc = f_overlap(frame_x,frame_y)
    baseline_jaccard = np.round(baseline_jac*100,2)
    baseline_dice = np.round(baseline_dc*100,2)
    
    return jaccard, baseline_jaccard, dice, baseline_dice


def f_diff_subsequent_t_v2(past_frames,future_frames,pred_future_frames,samp,ii,pred_save_dir):
    
    jaccard = np.zeros(3)
    dice = np.zeros(3)
   
    fig, axes = plt.subplots(2, 3, figsize=(8, 8))
    for idx in np.arange(3):
        frame_x = past_frames[samp,idx,0,:,256:512].detach().cpu().numpy()
        frame_y = future_frames[samp,idx,0,:,256:512].detach().cpu().numpy()

        diff = frame_y - frame_x
        diff *= (diff > 0)

    #     pred_frame_x = pred_future_frames[samp,idx,0,:,256:512].detach().cpu().numpy()
        pred_frame_y = pred_future_frames[samp,idx,0,:,256:512].detach().cpu().numpy()

        pred_diff = pred_frame_y - frame_x
        pred_diff *= (pred_diff > 0)

        intersect, union, jac, dc = f_overlap(diff,pred_diff)
        jaccard[idx] = np.round(jac*100,2)
        dice[idx] = np.round(dc*100,2)

        out = np.array(intersect + union)
        out = out - np.min(out)
        out = 255*out/np.max(out)
        
#         axes[int(idx),0].imshow(diff)
#         axes[int(idx),0].set_title(f"Frame {idx + 1}")
        
#         axes[int(idx),1].imshow(pred_diff)
#         axes[int(idx),1].set_title(f"Frame {idx + 1}")
       
        axes[0,int(idx)].imshow(f_get_mask(diff))
        axes[0,int(idx)].set_title(f"Groundtruth Diff, Frame {idx + 1}")
        
        axes[1,int(idx)].imshow(f_get_mask(pred_diff))
        axes[1,int(idx)].set_title(f"Predicted Diff, Frame {idx + 1}")
        
#         fig, ax = plt.subplots()
#         ax.imshow(intersect + union, cmap="gray")
# #         plt.title("Intersection and Union")
#         ax.axis("off")
#         plt.show()
               
        data = Image.fromarray(out.astype(np.uint8))
        save_name = pred_save_dir + "Growth/intersect_union" + "_" + str(int(ii)) + "_" + str(idx) + ".jpg"
        data.save(save_name)
    
    _, _, baseline_jac, baseline_dc = f_overlap(frame_x,frame_y)
    baseline_jaccard = np.round(baseline_jac*100,2)
    baseline_dice = np.round(baseline_dc*100,2)
    
    return jaccard, baseline_jaccard, dice, baseline_dice

def f_diff_subsequent_t_v3(past_frames,future_frames,pred_future_frames,samp,ii,pred_save_dir):
    
    jaccard = np.zeros(3)
    dice = np.zeros(3)
   
    fig, axes = plt.subplots(2, 3, figsize=(8, 8))
    for idx in np.arange(3):
        
        frame_x = past_frames[samp,idx,0,:,256:512]
        frame_x = 1*(frame_x > 0.5)
        frame_y = future_frames[samp,idx,0,:,256:512]
        frame_y = 1*(frame_y > 0.5)
        
#         pred_frame_x = pred_future_frames[samp,idx_t,0,:,256:512]
#         pred_frame_x = (pred_frame_x > 0.5)
        pred_frame_y = pred_future_frames[samp,idx,0,:,256:512]
        pred_frame_y = 1*(pred_frame_y > 0.5)

        diff = future_frames[samp,idx,0,:,512:]
#         diff = diff.detach().cpu().numpy()
        diff = (1*(diff>0.5)).detach().cpu().numpy()
#         diff2 = frame_y - frame_x
#         diff2 = (1*(diff2 > 0)).detach().cpu().numpy()
        
#         pred_diff = pred_future_frames[samp,idx_t+1,0,:,512:]
#         pred_diff = pred_diff.detach().cpu().numpy()
#         pred_diff = 1*((pred_diff>0.5)).detach().cpu().numpy()
        pred_diff2 =  pred_frame_y -  frame_x
        pred_diff2 = (1*(pred_diff2 > 0)).detach().cpu().numpy()
        
#         mask1 = f_get_mask(diff)
#         mask2 = f_get_mask(pred_diff)
#         _, _, jac, dc = f_overlap(diff, pred_diff)
        intersect, union, jac, dc = f_overlap(diff,pred_diff2)
#         _, _, jac, dc = f_overlap(diff2, pred_diff2)
        
        jaccard[idx] = jac*100
        dice[idx] = dc*100
#         jaccard[idx] = np.round(jac*100,2)
#         dice[idx] = np.round(dc*100,2)

        out = np.array(intersect + union)
        out = out - np.min(out)
        out = 255*out/np.max(out)
        
#         axes[int(idx),0].imshow(diff)
#         axes[int(idx),0].set_title(f"Frame {idx + 1}")
        
#         axes[int(idx),1].imshow(pred_diff)
#         axes[int(idx),1].set_title(f"Frame {idx + 1}")
       
        axes[0,int(idx)].imshow(f_get_mask(diff))
        axes[0,int(idx)].set_title(f"Groundtruth Diff, Frame {idx + 1}")
        
        axes[1,int(idx)].imshow(f_get_mask(pred_diff2))
        axes[1,int(idx)].set_title(f"Predicted Diff, Frame {idx + 1}")
        
#         fig, ax = plt.subplots()
#         ax.imshow(intersect + union, cmap="gray")
# #         plt.title("Intersection and Union")
#         ax.axis("off")
#         plt.show()
               
        data = Image.fromarray(out.astype(np.uint8))
        save_name = pred_save_dir + "Growth/intersect_union" + "_" + str(int(ii)) + "_" + str(idx) + ".jpg"
        data.save(save_name)
    
    _, _, baseline_jac, baseline_dc = f_overlap(frame_x,frame_y)
    baseline_jaccard = np.round(baseline_jac*100,2)
    baseline_dice = np.round(baseline_dc*100,2)
    
    return jaccard, baseline_jaccard, dice, baseline_dice

# def f_diff_subsequent_t_v3(future_frames,pred_future_frames,samp,ii,pred_save_dir):
    
#     jaccard = np.zeros(2)
#     dice = np.zeros(2)
   
#     fig, axes = plt.subplots(2, 2, figsize=(8, 8))
#     for idx in np.arange(2):
#         frame_x = future_frames[samp,idx,0,:,256:512].detach().cpu().numpy()
#         frame_y = future_frames[samp,idx+1,0,:,256:512].detach().cpu().numpy()

#         diff = future_frames[samp,idx+1,0,:,512:].detach().cpu().numpy()
#         pred_diff = pred_future_frames[samp,idx+1,0,:,512:].detach().cpu().numpy()

#         intersect, union, jac, dc = f_overlap(diff, pred_diff)
#         jaccard[idx] = np.round(jac*100,2)
#         dice[idx] = np.round(dc*100,2)
        
#         out = np.array(intersect + union)
#         out = out - np.min(out)
#         out = 255*out/np.max(out)
       
#         axes[0,int(idx)].imshow(f_get_mask(diff))
#         axes[0,int(idx)].set_title(f"Groundtruth Diff, Frame {idx + 1}")
        
#         axes[1,int(idx)].imshow(f_get_mask(pred_diff))
#         axes[1,int(idx)].set_title(f"Predicted Diff, Frame {idx + 1}")
               
#         data = Image.fromarray(out.astype(np.uint8))
#         save_name = pred_save_dir + "Growth/intersect_union" + "_" + str(int(ii)) + "_" + str(idx) + ".jpg"
#         data.save(save_name)
    
#     _, _, baseline_jac, baseline_dc = f_overlap(frame_x,frame_y)
#     baseline_jaccard = np.round(baseline_jac*100,2)
#     baseline_dice = np.round(baseline_dc*100,2)
    
#     return jaccard, baseline_jaccard, dice, baseline_dice

def f_table(overlap,overlap_subsequent_t,baseline_overlap,mode=0):
    if mode == 0:
        label0 = "Jaccard"
    else:
        label0 = "Dice"

    label1 = label0 + " of t3"
    label2 = label0 + " of t1 - t0"
    label3 = label0 + " of t2 - t1"
    label4 = label0 + " of t3 - t2"

    arr0 = np.append(("Sample"),np.arange(len(overlap))+1)
    arr1 = np.append(label1,overlap)
    arr2 = np.append(("Baseline"),baseline_overlap)
    arr3 = np.append((label2),overlap_subsequent_t[0,:])
    arr4 = np.append((label3),overlap_subsequent_t[1,:])
    arr5 = np.append((label4),overlap_subsequent_t[2,:])

    print(tabulate(list(zip(arr0,arr1,arr2,arr3,arr4,arr5)),tablefmt='fancy_grid',headers='firstrow'))

#     plt.plot(overlap, label = label1, color='b')
#     plt.plot(baseline_overlap, label = 'Baseline', color='r')
    plt.plot(overlap_subsequent_t[0,:], label = label2, color='b')
    plt.plot(overlap_subsequent_t[1,:], label = label3, color='r')
    plt.plot(overlap_subsequent_t[2,:], label = label4, color='g')
    
#     plt.fill_between(np.arange(np.shape(overlap)[0]), overlap-np.std(overlap), overlap+np.std(overlap), color='b', alpha=0.2)
#     plt.fill_between(np.arange(np.shape(overlap)[0]), baseline_overlap - np.std(baseline_overlap), baseline_overlap + np.std(baseline_overlap), color='r', alpha=0.2)
    plt.fill_between(np.arange(np.shape(overlap)[0]), overlap_subsequent_t[0,:] - np.std(overlap_subsequent_t[0,:]), overlap_subsequent_t[0,:] + np.std(overlap_subsequent_t[0,:]),color='b', alpha=0.2)
    plt.fill_between(np.arange(np.shape(overlap)[0]), overlap_subsequent_t[1,:] - np.std(overlap_subsequent_t[1,:]), overlap_subsequent_t[1,:] + np.std(overlap_subsequent_t[1,:]),color='r', alpha=0.2)
    plt.fill_between(np.arange(np.shape(overlap)[0]), overlap_subsequent_t[2,:] - np.std(overlap_subsequent_t[2,:]), overlap_subsequent_t[2,:] + np.std(overlap_subsequent_t[2,:]),color='r', alpha=0.2)
    
    plt.legend()
    plt.grid()
    plt.show()

#     plt.plot(overlap - baseline_overlap, label = label0 + ' - Baseline', color='b')
    
#     plt.fill_between(np.arange(np.shape(overlap)[0]), overlap - baseline_overlap-np.std(overlap - baseline_overlap), overlap - baseline_overlap+np.std(overlap - baseline_overlap), color='b', alpha=0.2)
#     plt.legend()
#     plt.grid()
#     plt.show()

    print('Mean ' + label0 + ':', round(np.mean(overlap),2), '%')
#     print('Mean ' + label0 +  ' of t2 - t1:', round(np.mean(overlap_subsequent_t[0,:]),2), '%')
#     print('Mean ' + label0 +  ' of t3 - t2:', round(np.mean(overlap_subsequent_t[1,:]),2), '%')
    print('Mean ' + label0 +  ' of t1 - t0:', round(np.mean(overlap_subsequent_t[0,:]),2), '%')
    print('Mean ' + label0 +  ' of t2 - t1:', round(np.mean(overlap_subsequent_t[1,:]),2), '%')
    print('Mean ' + label0 +  ' of t3 - t2:', round(np.mean(overlap_subsequent_t[2,:]),2), '%')
#     print('Mean improvement:', round(np.mean(overlap - baseline_overlap),2), '%')
    
def f_eval(pred_save_dir,sampler,VPTR_Transformer,VPTR_Enc,VPTR_Dec,N,device):

    jaccard = np.zeros(len(sampler)*N)
    baseline_jaccard = np.zeros(len(sampler)*N)
#     jaccard_subsequent_t = np.zeros((2,len(sampler)*N))
    jaccard_subsequent_t = np.zeros((3,len(sampler)*N))
    dice = np.zeros(len(sampler)*N)
    baseline_dice = np.zeros(len(sampler)*N)
#     dice_subsequent_t = np.zeros((2,len(sampler)*N))
    dice_subsequent_t = np.zeros((3,len(sampler)*N))

    for i in tqdm(np.arange(len(sampler))):
    # for i in np.arange(2)+12:

        try:
            sample = next(sampler)
        except StopIteration:
            break

        past_frames, future_frames, pred_future_frames = f_get_predictions(VPTR_Transformer,VPTR_Enc,VPTR_Dec,sample,device)

        for samp in np.arange(pred_future_frames.shape[0]):
            
            print("--------------------------------------------")
            print("--------------------------------------------")
            print("SAMPLE:", i*N+samp+1)
            
#             print("GA")
            print("Past Frames (Ground Truth):")
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for idx, ax in enumerate(axes.flat):

                x = past_frames[samp,idx,0,:,:].detach().cpu().numpy()

                ax.imshow(x)

                ax.set_title(f"Frame {idx + 1}")
            plt.show()
        
            print("Future Frames (Ground Truth):")
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for idx, ax in enumerate(axes.flat):

                x = future_frames[samp,idx,0,:,:].detach().cpu().numpy()

                ax.imshow(x)

                ax.set_title(f"Frame {idx + 1}")
            plt.show()

            print("Prediction:")
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for idx, ax in enumerate(axes.flat):

                y = pred_future_frames[samp,idx,0,:,:].detach().cpu().numpy()

                ax.imshow(y)

                ax.set_title(f"Frame {idx + 1}")
            plt.show()
            
           #  print("----------- MASKS --------------")
#             print("Ground Truth:")
#             fig, axes = plt.subplots(1, 3, figsize=(14, 4))
#             for idx, ax in enumerate(axes.flat):

#                 x = future_frames[samp,idx,0,:,:].detach().cpu().numpy()
#                 maskx = f_get_mask(x)

#                 ax.imshow(maskx, cmap="gray")

#                 ax.set_title(f"Frame {idx + 1}")
#             plt.show()

#             print("Prediction:")
#             fig, axes = plt.subplots(1, 3, figsize=(14, 4))
#             for idx, ax in enumerate(axes.flat):

#                 y = pred_future_frames[samp,idx,0,:,:].detach().cpu().numpy()
#                 masky = f_get_mask(y)

#                 ax.imshow(masky, cmap="gray")

#                 ax.set_title(f"Frame {idx + 1}")
#             plt.show()
            
#             fig.show()

    #         print(int(i*N+samp+1))

    #         plt.imshow(diff, cmap="gray")
    #         plt.show()

            frame_x = future_frames[samp,2,0,:,256:512].detach().cpu().numpy()
            frame_x = 1*(frame_x > 0.5).detach().cpu().numpy()
            frame_y = pred_future_frames[samp,2,0,:,256:512]
            frame_y = 1*(frame_y > 0.5).detach().cpu().numpy()

#             frame_x = frame_x - np.min(frame_x)
#             frame_y = frame_y - np.min(frame_y)
#             frame_x = frame_x/np.max(frame_x)
#             frame_y = frame_y/np.max(frame_y)

#             diff = f_save_diff(frame_x,frame_y,i*N+samp+1,pred_save_dir)

            _, _, jac, dc = f_overlap(frame_x,frame_y)
            jaccard[i*N+samp] = np.round(jac*100,2)
            dice[i*N+samp] = np.round(dc*100,2)
            print("Jaccard Index:", jaccard[i*N+samp])
            print("Dice Index:", dice[i*N+samp])

#             jaccard_subsequent_t[:,i*N+samp], baseline_jaccard[i*N+samp], dice_subsequent_t[:,i*N+samp], baseline_dice[i*N+samp] = f_diff_subsequent_t(future_frames,pred_future_frames,samp,i*N+samp+1,pred_save_dir)
            jaccard_subsequent_t[:,i*N+samp], baseline_jaccard[i*N+samp], dice_subsequent_t[:,i*N+samp], baseline_dice[i*N+samp] = f_diff_subsequent_t_v3(past_frames,future_frames,pred_future_frames,samp,i*N+samp+1,pred_save_dir)
            
#             frame_xd = future_frames[samp,2,0,:,512:].detach().cpu().numpy()
#             frame_yd = pred_future_frames[samp,2,0,:,512:].detach().cpu().numpy()
#             _, _, jac_diff, dc_diff = f_overlap(frame_xd,frame_yd)
#             jaccard_subsequent_t[:,i*N+samp] = np.round(jac_diff*100,2)
#             dice_subsequent_t[:,i*N+samp] = np.round(dc_diff*100,2)
            
    return jaccard, jaccard_subsequent_t, baseline_jaccard, dice, dice_subsequent_t, baseline_dice