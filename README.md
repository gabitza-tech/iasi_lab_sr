# Few-Shot Speaker Recognition

![alt text](https://github.com/gabitza-tech/iasi_lab_sr/blob/main/few-shot.png?raw=true)

For this lab, we will explore the meta-learning concept of Few-Shot Learning applied in the field fo Speaker Recognition. Compared to traditional machine learning methods, we assume that we receive at inference time a sample with a class that has not been previously seen in training. In order to tackle this problem, several architectures have been developed to handle training on a very small set of samples. However, another paradigm is leveraging pre-trained models on large datasets without further fine-tuning.

# First part of the lab
In this laboratory, we will utilize the Speaker Recognition model ECAPA-TDNN in order to extract embeddings and predict the identity of speakers that were not seen at training time. Initially, we will use the embeddings extracted from the voxceleb1 dataset and 

https://colab.research.google.com/drive/1-dQ2e6C_-1sZbhp65g92asmpgRk7CyP4#scrollTo=Aqxw7osNrd4s

For the embeddings:

https://ctipub-my.sharepoint.com/:u:/g/personal/gabriel_pirlogeanu_upb_ro/EdrWlHwqIVBEnK8-bCczdX8BSN_N17_a1cEJn6QP8EfIIg?e=fPLtco
https://ctipub-my.sharepoint.com/:u:/g/personal/gabriel_pirlogeanu_upb_ro/Eahwi5LaNo5Aq4r-SA2W7kMBNSqo4wtGVZ5JNm_l5ZHiDA?e=clwo2b

# Second part of the lab

You can download the pretrained model from:

https://ctipub-my.sharepoint.com/:u:/g/personal/gabriel_pirlogeanu_upb_ro/EdFPXb4iTHRIpd-YR2lRtyUBPU18-5xV7EXfsI0QECzZ-g?e=XaAvQx

In the second part of the lab, we will use a set of recordings from which we will extract the embeddings and create the support set and then create a pipeline where we predict based on a new recording. We will need the following packages:

```
torch 1.10
Cython
scikit-learn
```

In order to install torch correctly, you will need to do the following steps (UNLESS YOU HAVE ALREADY INSTALLED TORCH):

```
sudo apt-get install libopenblas-base libopenmpi-dev
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O  torch-1.10.0-cp36-cp36m-linux_aarch64.whl
sudo apt install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

```

In order to save a recording using the audio module, you can utilise the function `get_mic_pipeline` from a previous lab session:

https://github.com/Vladimirescu/iasi-emotion/blob/3b5e79bcab872535c2194055bf332ff5f0081ea5/pipelines/get_pipeline.py#L36

#Embeddings extraction
In order to extract embedding we will use the following script: (the script will return a .pkl file like the one we used in the google colab file)

`python3 src/extract_embeddings --initial_model models/vox2.model --eval_list files_list.txt`

The `files_list.txt` will need the following format:
```
id0 /full_path/to/audio_file0.wav
id0 /full_path/to/audio_file1.wav
id1 /full_path/to/audio_file2.wav
id2 /full_path/to/audio_file3.wav
id2 /full_path/to/audio_file4.wav
id2 /full_path/to/audio_file5.wav
...
```


