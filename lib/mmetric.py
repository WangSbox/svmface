import os,pickle
# 训练并保存模型
def saveclf(clf,path,acc,c,ite,accthreshold,**kargs):
    if acc >= accthreshold:
        with open(os.path.join(path, 'clf_C_' + str(c)+'_inter_'+str(ite)+'_.pickle'), 'wb') as f:
                pickle.dump(clf, f)
    else: pass
def show_accuracy(pred,target,mode):
    num = 0
    for i in range(target.size):
        if pred[i] == target[i]:            num += 1
    acc = num/target.size
    print(acc,   mode)
def perf_measure(y_true, y_pred, threshold):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:            TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:            FP += 1
        if y_true[i] == 0 and y_pred[i] == 0:            TN += 1
        if y_true[i] == 1 and y_pred[i] == 0:            FN += 1
    return TP, FP, TN, FN
def measure(TP,FP,TN,FN,freq):
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    if freq%2==0:
        print('测试集的查准/精确率为:{:.4f},召回率为:{:.4f},正确率为:{:.4f}'.format(Precision, Recall, Accuracy))
    else:
        print('验证集的查准/精确率为:{:.4f},召回率为:{:.4f},正确率为:{:.4f}'.format(Precision, Recall, Accuracy))
    return Accuracy