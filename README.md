## W2V-E2E-Language-diarization

The following repository contains code to the papaer 'End to End Spoken Language Diarization with Wav2vec Embeddings' which has been accepted in Interspeech 2023.

## Overview

Most of the Code-switched language suffers from primary language biasness. This is due to the unavailibility of of sufficient secondary language data. Our approach uses the pretrained features from a W2V model instead of using x-vector initially, which provides a performance improvement of around 30.7% in terms of Jaccard error rate (JER) over the baseline x-vector-based E2E (X-E2E) framework. The performance is furthur improved by using finetuned features from the W2V model and and modifying the temporal aggregation strategy from statistical pooling to attention pooling. The Final performance achieved in terms of JER is 22.5, which provides a relative improvement of 38.8% and 62.6% over the standalone W2V fine-tuned and the baseline X-E2E framework, respectively. 

![wav2vec](https://github.com/jagabandhumishra/W2v-E2E-Language-diarization/assets/91369740/cc7f5493-4d78-4256-bf00-6f010154e69b)

In the above figure (a) W2V pretraining (b) W2V finetuning. We extract a 768 dimensional feature vector during the pretraining and finetuning stage and then Statistical and Attentional Pooling is performed as shown in (c) and (d)

## Installation

The installation and training the W2V model during the pretraining and finetuning stage is same as [fairseq-vakyansh](https://github.com/Open-Speech-EkStep/vakyansh-wav2vec2-experimentation) 

#### Running the code

* After extracting the features (pretrained/finetuned). Create a .tsv file containing the path to the .npy array and its corresponding labels. The .tsv file shold be in the following format.
```
path/to/numpy_array1.npy  SSEEGSSE...
path/to/numpy_array2.npy  SGEEGGSE...
path/to/numpy_array3.npy  SEEEGESE...
path/to/numpy_array4.npy  SSEEGGSE...
.
.
.
``` 
* To train a W2V-ES (Statistical Pooling) by using the following command
```
python WaveBase.py --savedir "/data/KLESLD/Final_checkpoints/WavBase/Telagu_pre" \
 --train "/data/KLESLD/Dataset_WV2/Telugu/Train/WAVData.tsv" \
 --test "/data/KLESLD/Dataset_WV2/Telugu/Dev/WAVData.tsv" \
 --seed 0 --device 0 --batch 32 --epochs 60 --lang 3 --model my_base_model --lr 0.0001 --maxlength 666 --lmbda 0.5 --fll "T"
```

* To train a W2V-EA (Attention Pooling) by using the following command
```
python WaveAttention.py --savedir "/data/KLESLD/Final_checkpoints/WavBase/Telagu_pre" \
 --train "/data/KLESLD/Dataset_WV2/Telugu/Train/WAVData.tsv" \
 --test "/data/KLESLD/Dataset_WV2/Telugu/Dev/WAVData.tsv" \
 --seed 0 --device 0 --batch 32 --epochs 60 --lang 3 --model my_attention_model --lr 0.0001 --maxlength 666 --lmbda 0.5 --fll "T"
```
##### Note: To train a Gujrati or Tamil pretrained/finetuned model change ```--fll``` to 'G' or 'T' accordingly.
