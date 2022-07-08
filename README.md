# PredRNN++ for Radar Return
## Environment
- Python 3.6.15
- Pytorch 1.10.1

## Data
Data path is  ```/home/tintin/predrnn_model/predrnn_pp_radar_return/radar_image```
 - Type : Video sequences, each consisting of 6 frames. Each frame is a gray scale image(64*64 pixels) of radar return.
 - Input :  the first 4 frames of a sequence.
 - Output :  the 5th frame of a sequence.  
 - In/Output Shape : [batch_size,seq_lengh,channel,height,width]
 
## Training
 * Parameter
    * batch_size  : 32
    * EPOCHS : 1000
    * lr : 0.001 

Run   ```main.py``` for training the model,
 ```
 $ python main.py
 ```
After training, check  ```./result/ ``` directory to get input and output images.
To get detailed process in every epoch, run tensorboard.
 ```
 tensorboard --logdir tensorboard-mse_radar --bind_all
 ```
 
 The path for the trained model is ```/home/tintin/predrnn_model/predrnn_pp_radar_return/best_model_state_mse_radar.bin ```
