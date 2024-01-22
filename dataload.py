import numpy as np
import pandas
import os
from sklearn.model_selection import train_test_split as split

def load(hp,folder,ratio,shuf):
    keypoints = {0:    "Nose",1:    "Neck",2:    "RShoulder",3:    "RElbow",4:    "RWrist",5:    "LShoulder",
                 6:    "LElbow",7:    "LWrist",8:    "MidHip", 9:    "RHip",10:   "RKnee",11:   "RAnkle",12:   "LHip",
                 13:   "LKnee",14:   "LAnkle",15:   "REye",16:   "LEye",17:   "REar",18:   "LEar",19:   "LBigToe",
                 20:   "LSmallToe",21:   "LHeel",22:   "RBigToe",23:   "RSmallToe",24:   "RHeel"}
    classes = ["boxing","drums","guitar","rowing","violin"]
    labels = [0,1,2,3,4]

    # column names for csv files
    header = []
    for k in keypoints.values():
        #print(k)
        #header.append(k)
        header.append(k + '_x')
        header.append(k + '_y')
        header.append(k + '_c')
    header.append('R_ANGLE_ELBOW')
    header.append('R_ANGLE_ARMPIT')
    header.append('L_ANGLE_ELBOW')
    header.append('L_ANGLE_ARMPIT')
    
    train = []
    tlabel = []

    files = os.listdir(folder)
    dtwfiles =[]
    dtwlabels =[]
    print("Fetching Train Data from the memory")
    for j in range(len(files)):

        data = pandas.read_csv(f'{folder}{files[j]}',names=header)
        data = data.drop(data.columns[[2, 5, 8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,59,62,65,68,71,74]], axis=1)
        data = data.to_numpy()
        data = np.nan_to_num(data)
        dtwfiles.append(data)
        dtwlabels.append(classes.index(files[j][9:-4]))
        for i in range(int(data.shape[0]/hp)):
            train.append( data[i * hp : (i*hp) + hp,:] )
            tlabel.append(classes.index(files[j][9:-4]))
        Rp = int(((j+1)/len(files))*100)
        gap = ' '
        sign = '='    
        print('\r','[',(sign*Rp)+str('>')+gap*(100-Rp),']',"{} %".format(Rp),end = "")
        
    xtrainN,xtestN,ytrainN,ytestN = split(np.array(train),np.array(tlabel),test_size=ratio,shuffle = shuf)
    print("\n Data loaded...")
    return xtrainN,xtestN,ytrainN,ytestN,dtwfiles,dtwlabels,header,classes

