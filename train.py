
import numpy as np
from data_preprocessing.reshape_tensor import reshape_patch,reshape_patch_back
import torch
import torchgeometry as tgm

def train(model,data_loader,optimizer,eta,delta,batch_size,patch_size,input_length,seq_length,device,loss_func_name,img_width,img_height):
    
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    test = True
    model = model.train()
    losses = []
    mse_losses = []
    ssim_losses = []
    img_losses = []
    accur_losses = []
    #device = "cpuß
  
    img_channel = 1
    save_num = 0

    for data in data_loader:

        if(data.shape[0]!=batch_size):
            continue

        data = data.to(device)
        if eta > delta:
            eta -= delta
        else:
            eta = 0.0
       
            
        random_flip = np.random.random_sample(
            (batch_size,seq_length-input_length-1))
        true_token = (random_flip < eta)
        #true_token = (random_flip < pow(base,itr))

        ones = np.ones((int(img_height/patch_size),
                        int(img_width/patch_size),
                        patch_size**2*img_channel))
        zeros = np.zeros((int(img_height/patch_size),
                        int(img_width/patch_size),
                        patch_size**2*img_channel))
        mask_true = []
        for i in range(batch_size):
            for j in range(seq_length-input_length-1):
                if true_token[i,j]:
                    mask_true.append(ones)
                else:
                    mask_true.append(zeros)
        mask_true = np.array(mask_true)
        mask_true = np.reshape(mask_true, (batch_size,
                                           seq_length-input_length-1,
                                           int(img_height/patch_size),
                                           int(img_width/patch_size),
                                           patch_size**2*img_channel))
        mask_true = torch.from_numpy(mask_true)
        mask_true = (mask_true.permute(0,1,4,2,3)).to(device)
        
        for flip in range(2):

            radar_input = reshape_patch(data,patch_size)
            outputs,_output,_target, loss, mse_loss, ssim_loss,img_loss, accur_loss = (model(radar_input.float(),mask_true.float(),loss_func_name,patch_size,device,test = test))
            test = False
            losses.append(loss.item())
            loss.backward()

            mse_losses.append(mse_loss.item())
            ssim_losses.append(ssim_loss.item())
            img_losses.append(img_loss.item())
            accur_losses.append(accur_loss.item())
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if flip == 0:
                data = torch.flip(data, [1])
            
#             _, preds = torch.max(_output, dim=1)      
#             for i in range(batch_size):
#                 if (preds[i]==1):
#                     if (_target[i]==1):
#                         tp +=1
#                     else:
#                         fp +=1
#                 else:
#                     if (_target[i]==1):
#                         fn +=1
#                     else:
#                         tn +=1
#             print('_output')
#             print(_output.shape)
#             print('preds')
#             print(preds.shape)
            for i in range(batch_size):
                if (_output[i]<=0.75).any():
                    if (_target[i]<=0.75).any():
                        tp +=1
                    else:
                        fp +=1
                else:
                    if (_target[i]<=0.75).any():
                        fn +=1
                    else:
                        tn +=1
            
        if(save_num==0):
            save_num += 1
            Outputs = reshape_patch_back(outputs,patch_size)
            Radar_target = reshape_patch_back(radar_input,patch_size)
                
        elif save_num <5 :
            save_num += 1
            Outputs = torch.cat((Outputs,reshape_patch_back(outputs,patch_size)),0)
            Radar_target = torch.cat((Radar_target,reshape_patch_back(radar_input,patch_size)),0)
                
    np.save('./result/train_p_'+loss_func_name, (((Outputs).cpu()).detach().numpy()))
    np.save('./result/train_t_'+loss_func_name, (((Radar_target).cpu()).detach().numpy()))
    
    return [np.mean(losses), eta, np.mean(mse_losses), np.mean(ssim_losses), [tp,fp,tn,fn] ,np.mean(img_losses),np.mean(accur_losses)] #itr###

def eval_model(model,data_loader,batch_size,patch_size,input_length,seq_length,best_loss,device,loss_func_name,img_width,img_height,test = False):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    krt = 0
    krf = 0
    kst = 0
    ksf = 0
    srt = 0
    srf = 0
    sprt = 0
    sprf = 0
    model = model.eval()
    losses = []
    mse_losses = []
    ssim_losses = []
    img_losses = []
    accur_losses = []
    img_channel = 1
    save_num = 0
    test = True
    with torch.no_grad():   
        for data in data_loader:
            if(data.shape[0]!=batch_size):
                continue
            data = data.to(device)
            
            radar_input = reshape_patch(data,patch_size)

            mask_true = torch.from_numpy(np.zeros((batch_size,
                              seq_length-input_length-1,
                              int(patch_size**2*img_channel),
                              int(img_height/patch_size),
                              int(img_width/patch_size))))
            mask_true = mask_true.to(device)
            
            outputs,_output, _target,loss, mse_loss, ssim_loss,img_loss, accur_loss = (model(radar_input.float(),
                                                        mask_true.float(),
                                                        loss_func_name, 
                                                        patch_size,
                                                        device,
                                                        test = test))
            test = False
            
            losses.append(loss.item())
            mse_losses.append(mse_loss.item())
            ssim_losses.append(ssim_loss.item())
            img_losses.append(img_loss.item())
            accur_losses.append(accur_loss.item())
            
            
            _, preds = torch.max(_output, dim=1)      

#             for pics in output:
#                 for pic 
#                 if (pic[31:33,31:33]<=0.75).any():
#                     label.append(1)
#                 else:
#                     label.append(0)
        
            for i in range(batch_size):
                if (_output[i]<=0.75).any():
                    if (_target[i]<=0.75).any():
                        if((data[0,3,0,31:33,31:33]<=0.75).any()):
                            krt += 1
                        else: srt +=1
                        
                        tp +=1
                    else:
                        if((data[0,3,0,31:33,31:33]<=0.75).any()):
                            sprf += 1
                        else: ksf +=1
                            
                        fp +=1
                else:
                    if (_target[i]<=0.75).any():
                        if((data[0,3,0,31:33,31:33]<=0.75).any()):
                            krf += 1
                        else: srf +=1
                        fn +=1
                    else:
                        if((data[0,3,0,31:33,31:33]<=0.75).any()):
                            sprt += 1
                        else: kst +=1
                        tn +=1
            
            if(save_num==0):
                outputs = reshape_patch_back(outputs,patch_size)
                radar_input = reshape_patch_back(radar_input,patch_size)
                save_num += 1
                Outputs = outputs
                Radar_target = radar_input
                
            elif save_num <5 :
                outputs = reshape_patch_back(outputs,patch_size)
                radar_input = reshape_patch_back(radar_input,patch_size)
                save_num += 1
                Outputs = torch.cat((Outputs,outputs),0)
                Radar_target = torch.cat((Radar_target,radar_input),0)
                
    if test is False :
        if best_loss > np.mean(losses):
            np.save('./result/val_p_'+loss_func_name, (((Outputs).cpu()).detach().numpy()))
            np.save('./result/val_t_'+loss_func_name, (((Radar_target).cpu()).detach().numpy()))
            print("current loss " + str(np.mean(losses)) + " < best loss " + str(best_loss))



    return [np.mean(losses), np.mean(mse_losses), np.mean(ssim_losses), [tp,fp,tn,fn],np.mean(img_losses),np.mean(accur_losses),[krt,krf,kst,ksf,srt,srf,sprt,sprf]]  #itr###

def check_flip(arr):
    start_raining = 0
    stop_raining = 0
    for seq in arr:
        if(seq[3]^seq[4]==1):
            if seq[4]==1:start_raining+=1
            else:stop_raining+=1
    return [start_raining,stop_raining]

