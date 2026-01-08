import os
import sys
import numpy as np
import scipy.io as sio
#import tensorflow as tf
#from tensorflow.keras.applications import DenseNet121 as convnet
#from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import AutoImageProcessor, ViTModel
from PIL import Image
import torch

dirname1=('complete_dataset_tcc/melanoma')
dirname2=('complete_dataset_tcc/naevus')
extension=('.jpg')
dir2save1=('complete_dataset_tcc/embedding_melanoma')
dir2save2=('complete_dataset_tcc/embedding_naevus')


#dirname1=('complete_dataset_pped/melanoma')
#dirname2=('complete_dataset_pped/naevus')
#extension=('.jpg')
#dir2save1=('complete_dataset_pped/vit_embedding_melanoma')
#dir2save2=('complete_dataset_pped/vit_embedding_naevus')

if not os.path.exists(dir2save1):
    os.makedirs(dir2save1)

if not os.path.exists(dir2save2):
    os.makedirs(dir2save2)


#=========================================================================



image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
#========================================================================

dirname = dirname1
#dirname = dirname2

#model = convnet(include_top=False, weights='imagenet', pooling='avg')

#Organizando e numerando as imagens do diretorio
def inumerate_dir(dirname, extension):
 img_names= sorted(os.listdir(dirname))
 num_imgs= len(img_names)
 renamed_file=[]
 for i,img_name in enumerate(img_names, start=1):
  img_rename= '{:03d}{:s}'.format(i, extension)
  img_path= os.path.join(dirname,img_name)
  img_rename_path= os.path.join(dirname,img_rename)
  os.rename(img_path, img_rename_path)
  renamed_file.append(img_rename)

 return renamed_file

#Criando as features(vetor caracteristica)
def create_feature(dirname,dir2save, renamed_file):
 
 for num, indv_img in enumerate(renamed_file, start=1):
  #img = image.load_img(os.path.join(dirname,indv_img, target_size=(224, 224)))
  img=os.path.join(dirname, indv_img)
# x = image.img_to_array(img)
 # x = tf.expand_dims(x, axis=0)
 # x = preprocess_input(x)
 # features = model.predict(x)
#====================================================================================
  image = Image.open(img)
  inputs = image_processor(image, return_tensors="pt")

  with torch.no_grad():

    outputs = model(**inputs)
  features= outputs.last_hidden_state

#  last_hidden_states = outputs.last_hidden_state
#list(last_hidden_states.shape)
#===================================================================================

 # salvando a imagem
  if num == 1:
      mat = features
  else:
      mat = np.concatenate((mat, features), axis=0)
  
  path_dir = os.path.join(dir2save, 'vit_features.mat')
  data_save = {'features' : mat}
  sio.savemat(path_dir, data_save)
  
 

#sys.exit()
#renamed_file= inumerate_dir(dirname1, extension)
#create_feature(dirname1,dir2save1, renamed_file)

renamed_file= inumerate_dir(dirname2, extension)
create_feature(dirname2,dir2save2, renamed_file)

#features_melanoma=sio.loadmat(dir2save1 + '/features.mat')
#features_melanoma=features_melanoma['features']
#features_naevus=sio.loadmat(dir2save2 + '/features.mat')
#features_naevus=features_naevus['features']
