import os
import numpy as np
import scipy.io as sio
from PIL import Image
import torch

# Função para renomear arquivos
def inumerate_dir(dirname, extension):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    img_names = sorted(os.listdir(dirname))
    renamed_file = []
    for i, img_name in enumerate(img_names, start=1):
        img_rename = '{:03d}{:s}'.format(i, extension)
        img_path = os.path.join(dirname, img_name)
        img_rename_path = os.path.join(dirname, img_rename)
        os.rename(img_path, img_rename_path)
        renamed_file.append(img_rename)
    return renamed_file

# Função para criar features
def create_feature(name,image_processor,model,dirname, dir2save, renamed_file):
    for num, indv_img in enumerate(renamed_file, start=1):
        img = os.path.join(dirname, indv_img)
        image = Image.open(img).convert('RGB')
        inputs = image_processor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
        features = outputs.last_hidden_state.cpu().numpy()  # Convertendo para NumPy

        if num == 1:
            mat = features
        else:
            mat = np.concatenate((mat, features), axis=0)

        path_dir = os.path.join(dir2save, name)
        sio.savemat(path_dir, {'features': mat})


