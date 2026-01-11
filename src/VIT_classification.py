import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def classification_vit(features_melanoma, features_naevus):
    # --- (1) Garantir 2D: achatamento condicional (amostras x atributos) ---
    if features_melanoma.ndim > 2:
        features_melanoma = features_melanoma.reshape(features_melanoma.shape[0], -1)
    if features_naevus.ndim > 2:
        features_naevus = features_naevus.reshape(features_naevus.shape[0], -1)

    melanoma_train, melanoma_teste = train_test_split(features_melanoma, test_size=0.3, random_state=10)
    naevus_train,   naevus_teste   = train_test_split(features_naevus,   test_size=0.3, random_state=10)

    melanoma_teste, melanoma_validation = train_test_split(melanoma_teste, test_size=0.3, random_state=10)
    naevus_teste,   naevus_validation   = train_test_split(naevus_teste,   test_size=0.3, random_state=10)

    linhas = len(melanoma_train)
    label1 = np.ones(linhas)
    linhas = len(naevus_train)
    label2 = np.zeros(linhas)
    xlabel = np.concatenate((label1, label2), axis=0)
    xtrain = np.concatenate((melanoma_train, naevus_train), axis=0)

    linhas = len(melanoma_teste)
    label1 = np.ones(linhas)
    linhas = len(naevus_teste)
    label2 = np.zeros(linhas)
    ylabel = np.concatenate((label1, label2), axis=0)
    yteste = np.concatenate((melanoma_teste, naevus_teste), axis=0)

    linhas = len(melanoma_validation)
    label1 = np.ones(linhas)
    linhas = len(naevus_validation)
    label2 = np.zeros(linhas)
    zlabel = np.concatenate((label1, label2), axis=0)
    zvalidation = np.concatenate((melanoma_validation, naevus_validation), axis=0)

    scaler = StandardScaler()
    xtrain      = scaler.fit_transform(xtrain)
    yteste      = scaler.transform(yteste)
    zvalidation = scaler.transform(zvalidation)

    Kernel = 'rbf'
    Co    = [0.01, 0.1, 1, 10, 100]
    Gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    results_SVM = np.zeros((len(Co), len(Gamma)))

    for c_idx, c in enumerate(Co):
        for gam_idx, gam in enumerate(Gamma):
            clf = svm.SVC(C=c, kernel=Kernel, gamma=gam, probability=True)
            clf.fit(xtrain, xlabel)
            SVMproba = clf.predict_proba(zvalidation)
            SVMauc = roc_auc_score(zlabel, SVMproba[:,-1])
            results_SVM[c_idx, gam_idx] = SVMauc

    posmax_SVM = results_SVM.argmax()
    lmax = posmax_SVM // len(Gamma)
    cmax = posmax_SVM % len(Gamma)
    Copt = Co[lmax]
    Gammaopt = Gamma[cmax]

    N_neighbros = [1, 3, 5, 7, 9]
    results_KNN = np.zeros(len(N_neighbros))

    for n_idx, n in enumerate(N_neighbros):
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(xtrain, xlabel)
        KNNproba = knn.predict_proba(zvalidation)
        KNNauc = roc_auc_score(zlabel, KNNproba[:,-1])
        results_KNN[n_idx] = KNNauc

    posmax_KNN = results_KNN.argmax()

    nb = GaussianNB()
    nb.fit(xtrain, xlabel)

    NBproba = nb.predict_proba(zvalidation)
    NBauc = roc_auc_score(zlabel, NBproba[:,-1])

    lda = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, n_components=1)
    lda.fit(xtrain, xlabel)
    LDAproba = lda.predict_proba(zvalidation)
    LDAauc = roc_auc_score(zlabel, LDAproba[:,-1])

    svm_test = svm.SVC(C=Copt, kernel=Kernel, gamma=Gammaopt, probability=True)
    svm_test.fit(xtrain, xlabel)
    SVMproba_test = svm_test.predict_proba(yteste)
    SVMauc_test = roc_auc_score(ylabel, SVMproba_test[:,-1])
    SVMacc_test = accuracy_score(ylabel, svm_test.predict(yteste))

    knn_test = KNeighborsClassifier(n_neighbors=N_neighbros[posmax_KNN])
    knn_test.fit(xtrain, xlabel)
    KNNproba_test = knn_test.predict_proba(yteste)
    KNNauc_test = roc_auc_score(ylabel, KNNproba_test[:,-1])
    KNNacc_test = accuracy_score(ylabel, knn_test.predict(yteste))

    NBproba_test = nb.predict_proba(yteste)
    NBauc_test = roc_auc_score(ylabel, NBproba_test[:,-1])
    NBacc_test = accuracy_score(ylabel, nb.predict(yteste))

    LDAproba_test = lda.predict_proba(yteste)
    LDAauc_test = roc_auc_score(ylabel, LDAproba_test[:,-1])
    LDAacc_test = accuracy_score(ylabel, lda.predict(yteste))
