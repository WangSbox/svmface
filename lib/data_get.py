from  lib.ituils.ituils import *
import sklearn.model_selection as select
# from sklearn.model_selection import cross_val_score
def get_data(featurenums):
    
    real = np.empty([0,featurenums])
    path = '../data/traindataset/live'
    for file in os.listdir(path)[:]:
        # print(file)
        real = np.concatenate((real,np.load(os.path.join(path,file))[:,:]),axis=0)
    x_train,x_test = select.train_test_split(real,random_state=327,train_size=0.75,test_size=0.25)
    print(real.shape)
    del real

    fake =  np.empty([0,featurenums])
    path = '../data/traindataset/spoof'
    for file in os.listdir(path)[:]:
        # print(file)
        fake = np.concatenate((fake,np.load(os.path.join(path,file))[:,:]),axis=0)
    x_train1,x_test1 = select.train_test_split(fake,random_state=327,train_size=0.75,test_size=0.25)
    print(fake.shape)
    del fake

    # real = np.load('live5.npy')
    # fake = np.load('spoof5.npy')
    '''
    下面是新增数据
    '''
    real1 = np.empty([0,featurenums])
    path = '../data/traindataset1/live'
    for file in os.listdir(path):
        real1 = np.concatenate((real1,np.load(os.path.join(path,file))[:,:]),axis=0)
    x_train2, x_test2 = select.train_test_split(real1,random_state=327,train_size=0.75,test_size=0.25)
    print(real1.shape)
    del real1

    fake1 =  np.empty([0,featurenums])
    path = '../data/traindataset1/spoof'
    for file in os.listdir(path):
        fake1 = np.concatenate((fake1,np.load(os.path.join(path,file))[:,:]),axis=0)
    x_train3, x_test3 = select.train_test_split(fake1,random_state=327,train_size=0.6,test_size=0.4)
    print(fake1.shape)
    del fake1

    real_train = np.concatenate((x_train,x_train2),axis=0)
    fake_train = np.concatenate((x_train1,x_train3),axis=0)
    real_test = np.concatenate((x_test,x_test2),axis=0)
    fake_test = np.concatenate((x_test1,x_test3),axis=0)
    del x_train,x_train1,x_train2,x_train3,x_test,x_test1,x_test2,x_test3

    # real_train = x_train2
    # fake_train = x_train3
    # real_test = x_test2
    # fake_test = x_test3

    ra, rb, rc, rd = real_train.shape[0], fake_train.shape[0], real_test.shape[0], fake_test.shape[0]

    x_train = np.concatenate((real_train,fake_train),axis=0)
    x_test = np.concatenate((real_test,fake_test),axis=0)
    del real_train,fake_train,real_test,fake_test

    print(x_train.shape)
    y_train = np.zeros((ra + rb, 1))
    y_train[:ra] = 1

    y_test = np.zeros((rc + rd, 1))
    y_test[:rc] = 1
    return x_train,y_train,x_test,y_test