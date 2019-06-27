import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing, svm
from multiprocessing.pool import ThreadPool
from svm import Kernel, SVMTrainer
from load_dataset import get_CIFAR10_data
from preprocessing import preprocess_all
import gc

# %matplotlib inline
kernel="rbf"
lib=False

def train(x_train,y_train,c,gamma, comp = 176):

    x_train = preprocess_all(x_train)
    
    scaler = preprocessing.StandardScaler()

    # Fit on training set only.
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    pca = PCA(n_components= comp ).fit(x_train)
    x_pca = pca.transform(x_train)
    print("PCA components = ",x_pca.shape)
    if lib:
        if gamma==0:
            trainer = svm.SVC(C=c,kernel=kernel,gamma="scale")
        elif gamma < 0:
            trainer = svm.SVC(C=c,kernel=kernel,gamma="auto")
        else:
            trainer = svm.SVC(C=c,kernel=kernel,gamma=gamma)
    else:
        trainer=SVMTrainer(C=c,kernel=kernel,gamma=gamma)

    predictor=trainer.fit(x_pca,y_train)
    return predictor, scaler, pca

def explore_data(x_train,y_train):
    x_train = preprocess_all(x_train,y_train,True)
    
    scaler = preprocessing.StandardScaler()

    # Fit on training set only.
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    pca = PCA().fit(x_train)
    plt.figure(figsize=(30,15))
    plt.yticks(np.arange(0.0,1.01,0.05))
    plt.xticks(np.arange(0,257,16))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.grid()
    plt.savefig('pca.png')
    return scaler, np.argmax(np.cumsum(pca.explained_variance_ratio_)>0.9)

def test(x_test,y_test, predictor, scaler, pca):
    x_test = preprocess_all(x_test)
    x_test = scaler.transform(x_test)
    x_test = pca.transform(x_test)
    y_pred=predictor.predict(x_test)
    return 1.0-((np.count_nonzero(y_pred - y_test))/y_test.shape[0])

def performance(x_train,y_train,x_test,y_test,c,gamma):
    p,s,pca = train(x_train,y_train,c,gamma)
    return test(x_test,y_test, p, s, pca)

def try_parameters(x_train,y_train,x_val,y_val,Cs,gammas,pid):
    
    grid = np.zeros((Cs.shape[0],gammas.shape[0]))
    for l,c in enumerate(Cs):
        for m,gamma in enumerate(gammas):
            print(str(pid)+" prediction number "+str(l*Cs.shape[0]+m)+".... c="+str(c)+" gamma="+str(gamma))
            predictor,scaler,pca = train(x_train,y_train,c,gamma)
            y_pred = test(x_val,y_val, predictor, scaler,pca)
            print((y_pred.shape,y_val.shape))
            grid[l,m] = (float(np.count_nonzero(y_pred - y_val))) / y_val.shape[0]
            print("My score is "+str(grid[l,m]))
            print(str(pid)+" prediction number "+str(l*Cs.shape[0]+m)+" DONE!")
    return grid
                    
def cross_validation(Xtr,Ytr,Cs,gammas):
    grid = np.zeros((k,Cs.shape[0],gammas.shape[0]))
    block_sz = int(Xtr.shape[0]/k)
    pool = ThreadPool(processes=k)
    async_results = []
    for i in range(k):
        j = i*block_sz
        x_train = np.append(Xtr[:j],Xtr[j+block_sz:],axis=0)
        y_train = np.append(Ytr[:j],Ytr[j+block_sz:],axis=0)
        x_val = Xtr[j:j+block_sz]
        y_val = Ytr[j:j+block_sz]
        print("data split number ",i)
        async_results.append(pool.apply_async \
                (try_parameters,(np.copy(x_train),np.copy(y_train),np.copy(x_val),np.copy(y_val),Cs,gammas,i)))
    for i,res in enumerate(async_results):
        grid[i] = res.get()        
    np.save("full_grid",grid)
    np.save("average",np.mean(grid,axis=0))
    return grid

