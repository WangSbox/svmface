from  lib.ituils.ituils import *
import sklearn.svm as svm
from lib.mmetric import measure,perf_measure,show_accuracy,saveclf
from lib.data_get import get_data
from sklearn.preprocessing import StandardScaler,MinMaxScaler
def main():
    featurenums = 21858 ##特征长度
    x_train, y_train, x_test, y_test = get_data(featurenums)

    print('Begin')
    accthreshold = 0.5
    # C = [0.001,0.005,0.01,0.02,0.05,0.10,0.20,0.50,1,2,5,10]
    # inter = [50,100,200,260,400,500,700,1000]
    # kerl = ['poly','rbf']
    C = [50,100]
    inter = [5000,10000,50000]
    kerl = ['poly']
    freq = 0
    for kr in kerl:
        for c in C:
            for ite in inter:
                begin = time.time()
                print('C:{},kernel:{},inter:{}'.format(c,kr,ite))
                clf = svm.SVC(C=c,kernel=kr,gamma='auto',tol=0.01,cache_size=400,max_iter=ite,decision_function_shape='ovr',random_state=327)
                #clf = svm.SVC(C=1,kernel='rbf',probability=True,random_state=100,tol=1e-3)
                # score= cross_val_score(clf, x_train,y_train.ravel(), cv=5)
                clf.fit(x_train,y_train.ravel())
                y_hat = clf.predict(x_test)
                TP, FP, TN, FN = perf_measure(y_test.flatten(),y_hat.flatten(),0.1)
                acc = measure(TP, FP, TN, FN, freq)
                path = './modelv2/new2/'
                saveclf(clf,path,acc,c,ite,accthreshold)
                end = time.time()
                print('time cost:{:.4f}'.format(end-begin))
    return 0
if __name__ == "__main__":
    main()
    ## 训练时间增加，注意分配时间。