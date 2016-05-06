
# coding: utf-8

# In[1]:

#Modules
import os
import librosa
import h5py
import sklearn
import numpy as np


# ## Training Data 

# In[3]:

training_data_dir='/media/data_disk/datasets/TIMIT Dataset/timit/TIMIT/TRAIN/'


# ### Saving the directories for Training 

# In[4]:

f1=open('train_words.txt','w')
f2=open('train_phn.txt','w')
f3=open('train_wav.txt','w')
for root,directory,files in os.walk(training_data_dir):
    if len(files)==0:
        continue
    else:
        for i in files:
            #print i.split('.')[1]
            if i.split('.')[1]=='WAV':
                f3.write(str(root)+'/'+str(i)+'\n')
            elif i.split('.')[1]=='PHN':
                f2.write(str(root)+'/'+str(i)+'\n')
            elif i.split('.')[1]=='TXT':
                f1.write(str(root)+'/'+str(i)+'\n')

f1.close()
f2.close()
f3.close()


# ### Reading and Processing Train Phonemes File

# In[6]:

h5f=h5py.File('timit_files/train_phenome.h5','w')

phn_dict={}
with open('train_phn.txt','r') as f1:
    for path in f1:
        path=path.rstrip()
        word=path.split('/')[9]+'_'+path.split('/')[10].split('.')[0]
        phn_list=[]
        with open(path,'r') as f2:
            for phn in f2:
                phn_list.append(phn.split()[2])
            #phn_dict[word]=phn_list
            #print phn_list
            h5f.create_dataset(word,data=phn_list)
h5f.close()


# In[7]:

#remove the h#
#turns out you need the h#.....ignore this function/cell
'''
h5f=h5py.File('timit_files/train_phenome_cleaned.h5','w')

with open('audio_key.txt','r') as f:
    with h5py.File('timit_files/train_phenome.h5', 'r') as h5:
        for line in f:
            line=line.rstrip()
            #print line
            phn=h5[line][:]
            phn=np.delete(phn,0)
            phn=np.delete(phn,-1)
            h5f.create_dataset(line,data=phn)
h5f.close()
'''


# In[8]:

#extract all the phonemes types and store them into a list

#array that will hold all the phonemes
master_phn_list=[]

with open('audio_key.txt','r') as f:
    with h5py.File('timit_files/train_phenome.h5', 'r') as h5:
        for line in f:
            line=line.rstrip()
            phn=h5[line][:]
            for i in phn:
                if i not in master_phn_list:
                    master_phn_list.append(i)

                    
#add blank for ctc 
master_phn_list.append('_')

print master_phn_list
len(master_phn_list)


# In[9]:

#store the phenome master list array
h5f=h5py.File('timit_files/phoneme_list.h5','w')
h5f.create_dataset("list_phn",data=master_phn_list)
h5f.close()


# In[10]:

#insert spaces and encode the phn with a integer key to an array list
h5f=h5py.File('timit_files/phoneme_list_encode.h5','w')

with open('audio_key.txt','r') as f:
    with h5py.File('timit_files/train_phenome.h5', 'r') as h5:
        for line in f:
            line=line.rstrip()
            phn_space=[]
            phn=h5[line][:]
            for i in phn:
                phn_space.append(master_phn_list.index('_'))
                phn_space.append(master_phn_list.index(i))
            phn_space.append(master_phn_list.index('_'))
            h5f.create_dataset(line,data=phn_space)
h5f.close()


# In[11]:

#determine the largest phoneme 
max_length=0

with open('audio_key.txt','r') as f:
    with h5py.File('timit_files/phoneme_list_encode.h5', 'r') as h5:
        for line in f:
            line=line.rstrip()
            if len(h5[line][:])>max_length:
                max_length=len(h5[line][:])
print max_length


# In[12]:

#all phonemes are padded with spaces and of the same length
h5f=h5py.File('timit_files/phoneme_list_encode_space_padded.h5','w')

with open('audio_key.txt','r') as f:
    with h5py.File('timit_files/phoneme_list_encode.h5', 'r') as h5:
        for line in f:
            line=line.rstrip()
            phn=list(h5[line][:])
            phn_len=len(phn)
            if phn_len<max_length:
                for i in range(max_length-phn_len):
                    phn.append(master_phn_list.index('_'))
            h5f.create_dataset(line,data=phn)
h5f.close()


# ### Reading and Processing Train Word File

# In[13]:

h5f=h5py.File('timit_files/train_words.h5','w')
ctr=0

word_dict={}
with open('train_words.txt','r') as f1:
    for path in f1:
        path=path.rstrip()
        word=path.split('/')[9]+'_'+path.split('/')[10].split('.')[0]
        #print word
        word_list=[]
        with open(path,'r') as f2:
            for sent in f2:
                word_list.append(sent.split()[2:])
            #print word_list
            h5f.create_dataset(word,data=word_list)
h5f.close()

'''
# ### Reading and Processing Train Wav File 

# In[12]:

#reading audio files and converting them to numpy arrays
with h5py.File('timit_files/train_audio.h5', 'w') as h5:
    with open('train_wav.txt','r') as f1:
        for path in f1:
            path=path.rstrip()
            f_name=path.split('/')[9]+'_'+path.split('/')[10].split('.')[0]
            print f_name
            x,sr=librosa.load(path)
            mfcc_mat=librosa.feature.mfcc(x,sr,n_mfcc=55)
            scaled=sklearn.preprocessing.scale(mfcc_mat,axis=1)
            h5.create_dataset(f_name,data=scaled)


# In[14]:

#calculate the largest frame rate 
max_length=0
with open('audio_key.txt','r') as f:
    with h5py.File('timit_files/train_audio.h5', 'r') as h5:
        for name in f:
            name=name.rstrip()
            audio_arr=h5[name][:]
            if max_length<audio_arr.shape[1]:
                max_length=audio_arr.shape[1]
print max_length


# In[15]:

h5f=h5py.File('timit_files/train_audio_zero_padded.h5','w')

zeros=np.zeros((55,1))
with open('audio_key.txt','r') as f:
    with h5py.File('timit_files/train_audio.h5', 'r') as h5:
        for name in f:
            name=name.rstrip()
            audio_arr=h5[name][:]
            if audio_arr.shape[1]<max_length:
                for i in range(max_length-audio_arr.shape[1]):
                    audio_arr=np.append(audio_arr,zeros,axis=1)
            h5f.create_dataset(name,data=audio_arr)
            
h5f.close()
'''

# ## Testing Data

# In[16]:

test_data_dir='/media/data_disk/datasets/TIMIT Dataset/timit/TIMIT/TEST'


# ### Saving the directories for Testing

# In[57]:

f1=open('timit_files/test_words.txt','w')
f2=open('timit_files/test_phn.txt','w')
f3=open('timit_files/test_wav.txt','w')
for root,directory,files in os.walk(test_data_dir):
    if len(files)==0:
        continue
    else:
        for i in files:
            if i.split('.')[1]=='WAV':
                f3.write(str(root)+'/'+str(i)+'\n')
            elif i.split('.')[1]=='PHN':
                f2.write(str(root)+'/'+str(i)+'\n')
            elif i.split('.')[1]=='TXT':
                f1.write(str(root)+'/'+str(i)+'\n')

f1.close()
f2.close()
f3.close()

h5f.close()


# ## Reading and Processing Testing Data 

# In[60]:

h5f=h5py.File('timit_files/test_phenome.h5','w')

with open('timit_files/test_phn.txt','r') as f1:
    for path in f1:
        path=path.rstrip()
        word=path.split('/')[8]+"_"+path.split('/')[9].split('.')[0]
        phn_list=[]
        with open(path,'r') as f2:
            for phn in f2:
                phn_list.append(phn.split()[2])
            h5f.create_dataset(word,data=phn_list)
h5f.close()


# In[ ]:



