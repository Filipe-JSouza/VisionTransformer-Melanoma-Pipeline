import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import cv2

# Função para carregar imagens de uma pasta
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))  # Redimensionar as imagens para um tamanho fixo
            #img = img.flatten()  # Achatar a imagem em um vetor
            images.append(img)
    return np.array(images)


def bootstrap_ci(data, n_bootstrap=10000, confidence=0.95):
    boot_means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_means, (1 + confidence) / 2 * 100)
    return np.mean(data), lower, upper



def classification_CNN(dirname1, dirname2):
    # Carregar dados das pastas
    melanoma_train = load_images_from_folder(dirname1)
    naevus_train = load_images_from_folder(dirname2)

    # Criar rótulos
    label_melanoma = np.ones(len(melanoma_train))
    label_naevus = np.zeros(len(naevus_train))

    # Combinar dados e rótulos
    xtrain = np.concatenate((melanoma_train, naevus_train), axis=0)
    xlabel = np.concatenate((label_melanoma, label_naevus), axis=0)

    # Inicializar KFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)

    # Inicializar arrays para armazenar os resultados
    svm_aucs = []
    knn_aucs = []
    lda_aucs = []
    nb_aucs = []

    # Escalar os dados
    model = VGG16(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))

    X_all = preprocess_input(xtrain)
    features_all = model.predict(X_all)

    # Dentro do loop
    for train_index, val_index in kf.split(features_all, xlabel):
        X_train_fold, X_val_fold = features_all[train_index], features_all[val_index]
        y_train_fold, y_val_fold = xlabel[train_index], xlabel[val_index]

        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold = scaler.transform(X_val_fold)

    #============================================================================
    #SVM
        svm_clf = svm.SVC(C=0.01, kernel='rbf', gamma=0.001, probability=True)
        svm_clf.fit(X_train_fold, y_train_fold)
        SVMpred_prob = svm_clf.predict_proba(X_val_fold)[:, 1] 
        SVMauc = roc_auc_score(y_val_fold, SVMpred_prob)
        svm_aucs.append(SVMauc)
    # KNN
        knn_clf = KNeighborsClassifier(n_neighbors=5)
        knn_clf.fit(X_train_fold, y_train_fold)
        KNNpred_prob = knn_clf.predict_proba(X_val_fold)[:, 1]
        KNNauc = roc_auc_score(y_val_fold, KNNpred_prob)
        knn_aucs.append(KNNauc)

    # Naive Bayes
        nb_clf = GaussianNB()
        nb_clf.fit(X_train_fold, y_train_fold)
        NBpred_prob = nb_clf.predict_proba(X_val_fold)[:, 1]
        NBauc = roc_auc_score(y_val_fold, NBpred_prob)
        nb_aucs.append(NBauc)

    # LDA
        lda_clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None)
        lda_clf.fit(X_train_fold, y_train_fold)
        LDApred_prob = lda_clf.predict_proba(X_val_fold)[:, 1]
        LDAauc = roc_auc_score(y_val_fold, LDApred_prob)
        lda_aucs.append(LDAauc)

    #============================================================================
    # Resultados
    print(f"SVM AUCs: {[round(auc, 3) for auc in svm_aucs]}")
    print(f"SVM mean AUC: {np.mean(svm_aucs):.3f}")

    print(f"KNN AUCs: {[round(acc, 3) for acc in knn_aucs]}")
    print(f"KNN mean accuracy: {np.mean(knn_aucs):.3f}")

    print(f"Naive Bayes AUCs: {[round(acc, 3) for acc in nb_aucs]}")
    print(f"Naive Bayes mean accuracy: {np.mean(nb_aucs):.3f}")

    print(f"LDA AUCs: {[round(acc, 3) for acc in lda_aucs]}")
    print(f"LDA mean accuracy: {np.mean(lda_aucs):.3f}")


    svm_mean, svm_lower, svm_upper = bootstrap_ci(svm_aucs)
    knn_mean, knn_lower, knn_upper = bootstrap_ci(knn_aucs)
    nb_mean, nb_lower, nb_upper = bootstrap_ci(nb_aucs)
    lda_mean, lda_lower, lda_upper = bootstrap_ci(lda_aucs)

    print(f"SVM mean AUC: {svm_mean:.3f} (95% CI bootstrap: {svm_lower:.3f} - {svm_upper:.3f})")
    print(f"KNN mean AUC: {knn_mean:.3f} (95% CI bootstrap: {knn_lower:.3f} - {knn_upper:.3f})")
    print(f"Naive Bayes mean AUC: {nb_mean:.3f} (95% CI bootstrap: {nb_lower:.3f} - {nb_upper:.3f})")
    print(f"LDA mean AUC: {lda_mean:.3f} (95% CI bootstrap: {lda_lower:.3f} - {lda_upper:.3f})")
