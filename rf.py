from sklearn import metrics
import numpy as np
def evaluate (x_test,y_test,model_rf):
    x_test = x_test.reshape(len(x_test),(x_test.shape[1]*x_test.shape[2]))
    y_pred = np.argmax(model_rf.predict((x_test)),axis = 1)
    rf_acc = round(metrics.accuracy_score(y_pred,y_test)*100,2)
    cm = np.zeros((5,5))
    for i in range(len(y_test)):
            cm[y_test[i]][y_pred[i]]+=1
            
    return rf_acc,cm
def predict (eval_files,model_rf):
    res =[]
    for i in range(305):
        r = []
        for j in range(len(eval_files[i])):    
            t = eval_files[i][j].reshape(1,(eval_files[i][j].shape[0]*eval_files[i][j].shape[1]))
            r.append(t)
        res.append(np.argmax(model_rf.predict(t)))
    return res    