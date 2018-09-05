import pandas as pd    
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

COLUMNS = ["DEPTH","DT","GR","LLD","LLS","PHIE","PHIT","RHOB","SW","TNPH","VWCL"]
FEATURES = ["DEPTH","DT","GR","LLD","LLS","RHOB","TNPH"]
LABELS = ["phit","phie","sw","vwcl"]
LABEL = "phit"
COLUMNS=[c.lower() for c in COLUMNS]
FEATURES=[c.lower() for c in FEATURES]

def read_logs():
    folder="D:\SoftwareWebApps\Python\GeoProjects\AI&ML\\"
    training_set = pd.read_csv(folder+"lasdata.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv(folder+"lasdata-predict.csv",skipinitialspace=True, \
                             skiprows=1, names=COLUMNS)
    return training_set,prediction_set

def get_log_traintest(LABEL='phit',xtransform=True,ytransform=True):

    # COLUMNS

    # folder=r'D:\Ameyem Office\Geoservices\Neeraj & me\log data\\'
    # df=pd.read_csv('lasdata.csv')
    # df=pd.read_csv('lasdata.csv',skipinitialspace=True, \
    #                          skiprows=1, names=COLUMNS)

    training_set,prediction_set=read_logs()
    # prediction_set
    # df.head()
    scaler_x=None
    scaler_y=None
    if(xtransform):
        scaler_x = StandardScaler()
        scaler_x.fit(training_set[FEATURES].values)  
        x_train=scaler_x.transform(training_set[FEATURES].values).astype('float32')
        x_test= scaler_x.transform(prediction_set[FEATURES].values).astype('float32')
    else:
        x_train=training_set[FEATURES].values.astype('float32')
        x_test= prediction_set[FEATURES].values.astype('float32')

    if(ytransform):
        scaler_y = StandardScaler()
        scaler_y.fit(prediction_set[LABEL].values.reshape(-1, 1))
        y_train=scaler_y.transform(training_set[LABEL].values.reshape(-1, 1)).astype('float32')
        y_test=scaler_y.transform(prediction_set[LABEL].values.reshape(-1, 1)).astype('float32')
    else:    
        y_train=training_set[LABEL].values.reshape(-1, 1)
        y_test=prediction_set[LABEL].values.reshape(-1, 1)

    return x_train, x_test, y_train, y_test, scaler_x, scaler_y
# prediction_set[LABEL].values,Y_train.reshape(1,-1 )

def get_labelized_logdata(n_labels,ref=0,LABEL='phit'):
    training_set,prediction_set=read_logs()
    scaler_x = StandardScaler()
    scaler_x.fit(training_set[FEATURES].values)
    x_train=scaler_x.transform(training_set[FEATURES].values).astype('float32')
#     y_train=scaler_y.transform(training_set[LABEL].values.reshape(-1, 1)).astype('float32')

    x_test= scaler_x.transform(prediction_set[FEATURES].values).astype('float32')
    y=np.append(training_set[LABEL].values,prediction_set[LABEL].values)
    if(ref==0):
        y_=pd.qcut(pd.DataFrame(y,columns=['y'])['y'], n_labels, labels=False)['y'].values
    else:
        y_=y
        y_[y_>=ref]=1
        y_[y_<1]=0
    y_train=y_[0:len(training_set[LABEL])]
    y_pred=y_[-len(prediction_set[LABEL]):]
        
    return x_train,x_test,y_train,y_pred



def return_digits(Y,n_labels):
    y_d=np.zeros([len(Y),n_labels])
    for i in range(len(y_d)):
        y_d[i][Y[i]]=1.0
    return y_d
def get_digitized_logdata(n_labels):
    x_train,x_test,y_train,y_pred=get_labelized_logdata(n_labels)
    return x_train,x_test,return_digits(y_train,n_labels),return_digits(y_pred,n_labels)