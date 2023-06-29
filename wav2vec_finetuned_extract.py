import os
import pathlib
import fairseq
import numpy as np
import pandas as pd
import soundfile as sf
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
#%%
"""
# In case of any path related issue
1. Load the model with torch.load
   path = torch.load('checkpoint_best.pt')
2. Change the path in the model to where the CLSRIL-23.pt checkpoint is located
   path['cfg']['model']['w2v_path']='path/where/the/checkpoint/is/located/CLSRIL-23.pt'
3. Save the new model
   torch.save(path, 'checkpoint_new.pt')
"""
cp_path = '/DATA/jagabandhu/LD/WAV2VEC/Wav2vec_models/Gujarat/checkpoint_best_gujrati.pt' ##Change the model path to where the model file is located for Telegu, Tamil or Gujrati
train = glob.glob('/DATA/jagabandhu/LD/WAV2VEC/Gujrati/PartB_Gujarati/Train/Audio/*.wav') ## Change accordingly to where the audio files are stored
#transcript_path = '/DATA/jagabandhu/LD/Displace/Pretrained_files/w2v_fine/Train_overlapduration_500ms_labels/WAVDATA_train.txt'
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])

class MS_Dataset(Dataset):
    def __init__(self):
        self.train = train
        self.utter = []

        self.fname = []
        #fileExt = r"*.wav"
        #list = pathlib.Path(self.train).glob(fileExt)
        
        for path in self.train:
            #if path.is_file(): 
             #full_path = path.absolute()
             #my_path = full_path.as_posix()
             fname = path.split('/')[-1]
             data, fs = sf.read(path)
             self.utter.append(data)
             self.fname.append(fname)
       
    def __getitem__(self,index):
        output = self.utter[index] 
        outfile = self.fname[index]  
        final_torch = torch.from_numpy(np.array(output)) 
        return final_torch, outfile
    
    def __len__(self):
        return len(self.utter)
    
torch.cuda.empty_cache()
device = torch.device("cuda:0")
dataset = MS_Dataset()
dataloader = DataLoader(dataset = dataset, batch_size = 1)

model = model[0]

#putting it on device
model = model.to(device)
mo = model.w2v_encoder
mod = mo.w2v_model

output = []
#output = torch.FloatTensor(output)

mysavepath = '/DATA/jagabandhu/LD/Displace/Pretrained_files/w2v_fine/Train_overlapduration/'
save_dir = mysavepath+'/train'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
model.eval()
for batch_id, (batch, fname) in enumerate(dataloader):
    fname = str(fname[0]).split('.wav')[0]

    train_data = batch.to(dtype=torch.float)
    train_data = train_data.to(device)
    with torch.no_grad():
        z = mod.forward(train_data, mask=False, features_only=True)
    mynpy = z['x'].data.cpu().numpy()
    
    mynpy1 = mynpy[:, -1:, :]
    mynpy2 = np.concatenate((mynpy, mynpy1), axis=1)

    """
    dim1 = mynpy.shape[1]
    res = int(dim1//10)*10
    mynpy = mynpy[:, :res, :]
    """
    new_save_path = save_dir+'/'+fname+'.npy'
    np.save(new_save_path, mynpy2)

    del z
    torch.cuda.empty_cache()
    del train_data
    torch.cuda.empty_cache()
    """
    with open(transcript_path) as file:
        tsv_file = file.readlines()
        with open(mysavepath+'/WAVData_TRAIN.txt','a') as my_txt:
            for line in tsv_file:
                a,b = line.split('\t')
                if(a == fname):
                    my_txt.write(save_dir+'/'+fname+'.npy\t'+b+'\n')
                    break
    """
# %%
