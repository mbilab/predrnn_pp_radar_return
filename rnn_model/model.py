from rnn_model.lstm import cslstm
from rnn_model.GradientHighwayUnit import gradient_highway as ghu
import torch
import torch.nn as nn
import torchgeometry as tgm
import numpy as np
import math
from os.path import exists
from data_preprocessing.reshape_tensor import reshape_patch,reshape_patch_back

class RNN(nn.Module):
    def __init__(self, input_shape, num_layers, num_hidden, seq_length,input_length,device , tln=True, loss_func=nn.MSELoss()):
        super(RNN,self).__init__()
        self.guassion = get_guassion_matrix()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.input_shape = input_shape
        
        filter_size = 5
        self.output_length = seq_length - input_length
        self.input_length = input_length
        self.gradient_highway = ghu('highway', 
                                    filter_size, 
                                    num_hidden[0], 
                                    input_shape[0], 
                                    input_shape[3], 
                                    input_shape[4], 
                                    device, 
                                    tln)
        self.conv = nn.Conv2d(self.num_hidden[-1], input_shape[2], 1, 1, 0) ###
        self.loss_func = loss_func
        self.lstm = []
        self.conv1 =  nn.Conv2d(1, 16, 4, stride=3)
        self.conv2 =  nn.Conv2d(16, 64, 3, stride=2)
        self.conv3 =  nn.Conv2d(64, 1, 3, stride=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 4)
        

        
        for i in range(num_layers):
            if i == 0:
                num_hidden_in = num_hidden[-1]
            else:
                num_hidden_in = num_hidden[i-1]
            
            input_channel = input_shape[2]
            new_cell = cslstm('lstm_'+str(i+1),
                  filter_size,
                  num_hidden_in,
                  num_hidden[i],
                  input_shape,
                  device,
                  input_channel,
                  tln=tln)
            self.lstm.append(new_cell)
     
            
        self.lstm = nn.ModuleList(self.lstm)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, mask_true,loss_func_name,patch_size, device, test = False):
        # [batch, length, channel, width, height]
        gen_images = []
        cell = []
        hidden = []
        guassion = (self.guassion).to(device)
        mem = None
        z_t = None 
        for i in range(self.num_layers):
            cell.append(None)
            hidden.append(None)
            
        for t in range(self.seq_length-1):
        
            if t < self.input_length:
                inputs = images[:,t]
            else:
                inputs = mask_true[:,t-self.input_length]*images[:,t] + (1-mask_true[:,t-self.input_length])*x_gen
            
            
            hidden[0], cell[0], mem = self.lstm[0](inputs, hidden[0], cell[0], mem) #?
  
            z_t = self.gradient_highway(hidden[0], z_t)
            hidden[1], cell[1], mem = self.lstm[1](z_t, hidden[1], cell[1], mem)

            for i in range(2, self.num_layers):
                hidden[i], cell[i], mem = self.lstm[i](hidden[i-1], hidden[i], cell[i], mem)
                
            x_gen = self.conv(hidden[self.num_layers-1])
            gen_images.append(x_gen)
        

        gen_images = self.sigmoid(torch.stack(gen_images,dim=1))
       
        l2_loss = nn.MSELoss().to(device)
        l1_loss = nn.L1Loss().to(device)
        ssim_func = tgm.losses.SSIM(5,reduction = 'mean')
        
        a = reshape_patch_back(gen_images,patch_size)
        b = reshape_patch_back(images[:,1:],patch_size)
        l1_guassion = (torch.mul(abs(a-b),guassion))
        l2_guassion = (torch.mul((torch.square(a-b)),guassion))
        
        loss = ((torch.sum(l1_guassion)) + (torch.sum(l2_guassion)))/ (torch.sum(guassion))
        img_loss =  torch.tensor(0)
        accur_loss = torch.tensor(0)
        
        x = a[:,-2:-1,0,31:33,31:33]
        y = b[:,-2:-1,0,31:33,31:33]
        
   
        output = gen_images[:,-(self.output_length):,:,:,:]
        target = (images[:,1:])[:,-(self.output_length):,:,:,:]
        
        mse_loss = l2_loss(output,target)
        ssim_loss = ssim_func((torch.reshape(output,(-1,(self.output_length),64,64))),(torch.reshape(target,(-1,(self.output_length),64,64))))
  
 
        return [gen_images,x,y , loss, mse_loss, ssim_loss, img_loss, accur_loss]

class FineTuning(nn.Module):
    def __init__(self, input_shape, num_layers, num_hidden, seq_length,input_length,device, tln=True, loss_func=nn.MSELoss()):
        super(FineTuning,self).__init__()
        self.guassion = get_guassion_matrix()
        self.predrnn = (RNN(input_shape, num_layers, num_hidden, seq_length,input_length, device, True, loss_func)).to(device)
        if(exists('best_model_state_mse_radar_pretrained.bin')):
            self.predrnn.load_state_dict(torch.load('best_model_state_mse_radar_pretrained.bin'))
        self.loss_func = loss_func
        
        
        self.conv1 =  nn.Conv2d(1, 16, 4, stride=1)
        self.conv2 =  nn.Conv2d(16, 64, 4, stride=1)
        #self.conv3 =  nn.Conv2d(256, 64, 3, stride=1)

        self.relu = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 4)
        
        self.conv4 =  nn.Conv2d(1, 8, 4, stride=3)
        self.conv5 =  nn.Conv2d(8, 16, 3, stride=2)
        self.conv6 =  nn.Conv2d(16, 1, 3, stride=1)
        
        self.ln = nn.Linear(5184, 4)
        self.ln2 = nn.Linear(4096,4)
        
        self.sig = nn.Sigmoid()
     
        self.conv11 =  nn.Conv2d(1, 16, 32, stride=1)
        
        self.conv12 = nn.Conv2d(1, 1, 32, stride=1)
        
        
        self.ln3 = nn.Linear(64*64,2)
        self.celoss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images, mask_true,loss_func_name,patch_size, device, test = False):
        
        gen_images,_,_,_, mse_loss, ssim_loss,_,_ = (self.predrnn(images.float(),mask_true.float(),loss_func_name,patch_size,device,test = False))
        guassion = (self.guassion).to(device)
        l2_loss = nn.MSELoss().to(device)
        l1_loss = nn.L1Loss().to(device)
        a = reshape_patch_back(gen_images,patch_size)
        b = reshape_patch_back(images[:,1:],patch_size)
        l1_guassion = (torch.mul(abs(a-b),guassion))
        l2_guassion = (torch.mul((torch.square(a-b)),guassion))
        img_loss = ((torch.sum(l1_guassion)) + (torch.sum(l2_guassion)))/ (torch.sum(guassion))
        
        x = a[:,-2:-1,0,31:33,31:33]
        y = b[:,-2:-1,0,31:33,31:33]
        accur_loss = (3000)*self.loss_func(x, y)
        loss = accur_loss + img_loss
        
        return [gen_images,x,y , loss, mse_loss, ssim_loss, img_loss, accur_loss]
        
def get_guassion_matrix():
    guassion = np.zeros([64,64])
    sigmaSpace = 0.3*((64-1)*0.5-1)+0.8
    for i in range(-31,33):
        for j in range(-31,33):
            l = i-0.5
            m = j-0.5
            guassion[i+31,j+31] = (math.exp(-(0.5*((l*l)+(m*m)))/(sigmaSpace*sigmaSpace)))/(2*math.pi*sigmaSpace*sigmaSpace)
    guassion = torch.from_numpy(guassion)
    return guassion

if __name__ == '__main__':
    a = torch.randn(3, 20, 1, 64, 64)
    shape = [3, 20, 1, 64, 64]
    numlayers = 4
    predrnn = RNN(shape, numlayers, [64,64,128,128], 20, True) #???
    predict, loss = predrnn(a)
    print(predict.shape)
