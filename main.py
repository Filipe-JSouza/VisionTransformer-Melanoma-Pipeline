import scipy.io as sio
import src.CNN_classification
import src.vit_feature_extract
import src.VIT_classification
from transformers import AutoImageProcessor, ViTModel

def main():
    dirname1 = r'data\\melanoma'
    dirname2 = r'data\\naevus'
    dirname3=r'\\tcc_2\data\\vit_features'
    src.CNN_classification.classification_CNN(dirname1,dirname2)

    # Carregando processador e modelo ViT
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    extension = ('.jpg')

    # Processando diretório
    renamed_file = src.VIT_classification.inumerate_dir(dirname1, extension)
    src.VIT_classification.create_feature('melanoma_features.mat',
                                          image_processor,model,dirname1,
                                            dirname3, renamed_file)

    # Processando diretório
    renamed_file = src.VIT_classification.inumerate_dir(dirname1, extension)
    src.VIT_classification.create_feature('naevus_features.mat',
                                          image_processor,model,dirname2,
                                            dirname3, renamed_file)
      
    features_melanoma = sio.loadmat(dirname3 + '\\melanoma_features.mat')
    features_melanoma = features_melanoma['features']
    features_naevus   = sio.loadmat(dirname3 + '\\naevus_features.mat')
    features_naevus   = features_naevus['features']
    src.vit_feature_extract.classification_vit(features_melanoma, features_naevus)
    
if __name__ == '__main__':
    main()