
import torch
import numpy as np
from train import train , eval_model
import time
from data_preprocessing import data
from sklearn.model_selection import train_test_split
from numpy import newaxis
from tensorboardX import SummaryWriter
import torchgeometry as tgm
import os
from rnn_model.model import RNN,FineTuning


start_time = time.time()
cuda = "cuda:1"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
finetuning = True # or pretraining

if(finetuning):
    loss_func_name = "mse_radar"
else:
    loss_func_name = "mse_radar_pretrained"

device = torch.device(cuda if torch.cuda.is_available() else "cpu")

print("Start training")

seq_length = 6
input_length = 4
height = 64
width = 64

print("Loading data")
radar_image = []


lr = 0.001
delta = 0.00002
base = 0.99998
eta = 1
EPOCHS = 1000

loss_function = torch.nn.MSELoss().to(device)
batch_size = 32
patch_size = 4

img_channel = 1


shape = [batch_size, seq_length, patch_size*patch_size*img_channel, int(height/patch_size), int(width/patch_size)]
num_hidden = [128,64,64,64]
numlayers = len(num_hidden)

if(finetuning):
    model = (FineTuning(shape, numlayers, num_hidden, seq_length,input_length, device, True, loss_function,)).to(device) 
else:
    model = (RNN(shape, numlayers, num_hidden, seq_length,input_length, device, True, loss_function,)).to(device) 

radar_train = np.load("./radar_image/train_data.npy")[:,:seq_length,np.newaxis,:,:]
radar_val = np.load("./radar_image/valid_data.npy")[:,:seq_length,np.newaxis,:,:]

print("Loading finish")
print('Train_Data_shape is ' + str(radar_train.shape))
print('Val_Data_shape is ' + str(radar_val.shape))

train_set = data.RadarDataset(data = radar_train)
val_set = data.RadarDataset(data = radar_val)

train_dataloader = data.DataLoader(train_set, batch_size=batch_size, shuffle = True)
val_dataloader = data.DataLoader(val_set, batch_size=batch_size ,shuffle = True)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
writer = SummaryWriter("tensorboard-"+loss_func_name)

best_loss = 10000

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('----------')
    train_loss, eta, train_mse, train_ssim , train_cfsmatrix, train_img_loss, train_accur_loss= train(model,train_dataloader,optimizer,eta,delta,batch_size,patch_size,
                                                                    input_length,seq_length,device,loss_func_name,width,height)
    tp,fp,tn,fn = train_cfsmatrix
    
    print("#Train")
    print(f'loss : {round(float(train_loss),7)} ')
    print(f'accuracy : {round(float((tp+tn)/(sum(train_cfsmatrix))),7)} ')
    if(finetuning):
        print(f'img loss : {round(float(train_img_loss),7)} ')
        print(f'accur loss : {round(float(train_accur_loss),7)} ')
        print(f'[tp,fp,tn,fn] {train_cfsmatrix }\n')
    
    val_loss, val_mse, val_ssim, val_cfsmatrix, val_img_loss, val_accur_loss,_= (eval_model
                                  (model,val_dataloader,batch_size,
                                   patch_size,input_length,seq_length,
                                   best_loss,device,loss_func_name,width,height))
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch+1
        torch.save(model.state_dict(), 'best_model_state_' + loss_func_name  + '.bin')
        
    _tp,_fp,_tn,_fn = val_cfsmatrix
    print("\n#Val")
    print(f'loss : {round(float(val_loss),7)}')
    print(f'accuracy : {round(float((_tp+_tn)/(sum(val_cfsmatrix))),7)} ')
    
    if(finetuning):
        print(f'img loss : {round(float(val_img_loss),7)} ')
        print(f'accur loss : {round(float(val_accur_loss),7)} ')
        print(f'[tp,fp,tn,fn] {val_cfsmatrix }\n')
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    writer.add_scalars("loss",{
    'train': np.asscalar(train_loss),
    'validation': np.asscalar(val_loss),
    }, epoch+1)
    writer.add_scalars("mse",{
    'train': np.asscalar(train_mse),
    'validation': np.asscalar(val_mse),
    }, epoch+1)
    writer.add_scalars("ssim",{
    'train': np.asscalar(train_ssim),
    'validation': np.asscalar(val_ssim),
    }, epoch+1)
    
    if(finetuning):
        writer.add_scalars("accur_loss",{
        'train': np.asscalar(train_accur_loss),
        'validation': np.asscalar(val_accur_loss),
        }, epoch+1)
        writer.add_scalars("train_img_loss",{
        'train': np.asscalar(train_img_loss),
        'validation': np.asscalar(val_img_loss),
        }, epoch+1)
    

    writer.add_scalars("accuracy",{
    'train': ((tp+tn)/(sum(train_cfsmatrix))),
    'validation': ((_tp+_tn)/(sum(val_cfsmatrix))),
    }, epoch+1)
    
    if (tp+fp)!= 0:
        pt = (tp)/(tp+fp)
    else:
        pt = 0
        
    if (_tp+_fp)!=0:
        pv = (_tp)/(_tp+_fp)
    else:
        pv = 0
    
    writer.add_scalars("precision(rain)",{
    'train': (pt),
    'validation': (pv),
    }, epoch+1)

    if (tn+fn)!= 0:
        pt = (tn)/(tn+fn)
    else:
        pt = 0
        
    if (_tn+_fn)!=0:
        pv = (_tn)/(_tn+_fn)
    else:
        pv = 0
                
    writer.add_scalars("precision(no rain)",{
    'train': (pt),
    'validation': (pv),
    }, epoch+1)
    
    
    if (tp+fn)!= 0:
        rt = (tp)/(tp+fn)
    else:
        rt = 0
        
    if (_tp+_fn)!=0:
        rv = (_tp)/(_tp+_fn)
    else:
        rv = 0  

    writer.add_scalars("recall(rain)",{
    'train': (rt),
    'validation': (rv),
    }, epoch+1)
        
    if (tn+fp)!= 0:
        rt = (tn)/(tn+fp)
    else:
        rt = 0
        
    if (_tn+_fp)!=0:
        rv = (_tn)/(_tn+_fp)
    else:
        rv = 0  

    writer.add_scalars("recall(no rain)",{
    'train': (rt),
    'validation': (rv),
    }, epoch+1)
    
    
    print("\n")


    
print("Best loss : " + str(best_loss))
print("Best epoch : " + str(best_epoch))
