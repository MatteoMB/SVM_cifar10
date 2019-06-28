from multiprocessing.pool import ThreadPool
import numpy as np

def try_parameters(x_train,y_train,x_val,y_val,Cs,gammas):
    grid = np.zeros((Cs.shape[0],gammas.shape[0]))
    for l,c in enumerate(Cs):
        for m,gamma in enumerate(gammas):
            print("Prediction number "+str(l*gammas.shape[0]+m)+" of "+str(Cs.shape[0]*gammas.shape[0]))
            predictor,scaler,pca = train(x_train,y_train,c,gamma)
            grid[l,m] = test(x_val,y_val, predictor, scaler,pca)
    return grid
                    
def cross_validation(Xtr,Ytr,Cs,gammas,k=5):
    grid = np.zeros((k,Cs.shape[0],gammas.shape[0]))
    print(str(k)+"-fold cross-validating of parameters\n C: ",Cs,"\nand gamma :",gammas)
    block_sz = int(Xtr.shape[0]/k)
    pool = ThreadPool(processes=k)
    async_results = []
    for i in range(k):
        j = i*block_sz
        x_train = np.append(Xtr[:j],Xtr[j+block_sz:],axis=0)
        y_train = np.append(Ytr[:j],Ytr[j+block_sz:],axis=0)
        x_val = Xtr[j:j+block_sz]
        y_val = Ytr[j:j+block_sz]
        print("data split number ",i,"......")
        async_results.append(pool.apply_async \
                (try_parameters,(np.copy(x_train),np.copy(y_train),np.copy(x_val),np.copy(y_val),Cs,gammas)))
    for i,res in enumerate(async_results):
        grid[i] = res.get()        
    np.save("full_grid",grid)
    av = np.mean(grid,axis=0)
    np.save("average",av)
    indexes = np.unravel_index(np.argmax(av, axis=None), av.shape)
    print("Best values are found in indexes ",indexes)
    return grid, indexes


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing, svm

#%matplotlib inline

def train(x_train,y_train,c,gamma, comp = 176):

    x_train = preprocess_all(x_train)
    
    scaler = preprocessing.StandardScaler()

    # Fit on training set only.
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    pca = PCA(n_components= comp ).fit(x_train)
    x_pca = pca.transform(x_train)
    kernel='rbf'
    if gamma==0:
        gamma='scale'
    elif gamma < 0:
        kernel='linear'
        gamma ='auto'
    if lib:
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
    plt.figure(figsize=(18,9))
    plt.yticks(np.arange(0.0,1.01,0.05))
    plt.xticks(np.arange(0,257,16))
    variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(variance_ratio)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.grid()
    plt.savefig('pca.png')
    n_comp = np.argmax(variance_ratio>0.9)
    pca = PCA(n_components=n_comp).fit(x_train)
    varx = np.var(pca.transform(x_train))
    print("variance of x_train: ",varx)
    return n_comp, varx

def test(x_test,y_test, predictor, scaler, pca):
    x_test = preprocess_all(x_test)
    x_test = scaler.transform(x_test)
    x_test = pca.transform(x_test)
    y_pred=predictor.predict(x_test)
    return 1.0-((np.count_nonzero(y_pred - y_test))/y_test.shape[0])

def performance(x_train,y_train,x_test,y_test,c,gamma):
    p,s,pca = train(x_train,y_train,c,gamma)
    return test(x_test,y_test, p, s, pca)


def main():
    x_train, y_train, x_test, y_test = get_CIFAR10_data()
    comp, varx = explore_data(x_train,y_train)
    x_train, y_train, x_test, y_test = get_CIFAR10_data(num_training=2000, num_test=2000)
    print('Train data shape: ', x_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Test data shape: ', x_test.shape)
    print('Test labels shape: ', y_test.shape)
    k=5
    cs = np.power(10,np.arange(-1,2).astype(np.double))
    g0 = 1/comp
    gammas = g0 * np.append([-1,0],np.power(10,np.arange(-2,2).astype(np.double)))
    grid,indexes = cross_validation(x_train,y_train,cs,gammas,k)
    print("accuracy: "performance(x_train,y_train,x_test,y_test,cs[indexes[0]],gammas[indexes[1]]))
    
def all_classes():
    res = np.zeros(100).reshape(10,10)
    print("Using default values compute the other classifiers")
    for i in range(10):
        for j in range(10):
            if i < j:
                x_train, y_train, x_test, y_test = get_CIFAR10_data(cl0=i,cl1=j,num_training=2000, num_test=2000)
                acc = performance(x_train,y_train,x_test,y_test,1,0)
                res[i,j] = res[j,i] = acc
                print(classes[i]," ",classes[j], " accuracy: ", acc)
    plt.figure(figsize=(10,4))
    plt.axis('tight')
    plt.axis('off')
    plt.table(cellText=np.round(res,2),colLabels=classes,rowLabels = classes ,loc='center')

if __name__ == "__main__":
    lib = False
    main()
    lib = True
    all_classes()