def plot_pca():    
    # -*- coding: utf-8 -*-
    """
    Created on Sun Jan 15 20:16:29 2023

    @author: ashan
    """
    import pandas as pd
    import numpy as np
    import random as rd
    from sklearn.decomposition import PCA
    from sklearn import preprocessing
    import matplotlib.pyplot as plt
    import os
    class VisualizeScatter:
        def __init__(self, fig_size=(10, 8), xlabel='X', ylabel='Y', title=None, 
                     size=10, num_classes=5):
            plt.figure(figsize=fig_size)
            plt.grid('true')
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            self.colors = ['red', 'green', 'blue','yellow','violet']
            self.num_classes = num_classes
            self.size = size
     
        def add_scatters(self, X):
            x = X[:, 0]
            if X.shape[1] == 2:
                y = X[:, 1]
            else:
                y = np.zeros(len(x))
            points_per_class = len(x) // self.num_classes
            st = 0
            end = points_per_class
            for i in range(self.num_classes):
                plt.scatter(x[st:end], y[st:end], 
                    c=self.colors[i % len(self.colors)], 
                    s=self.size)
                st = end
                end = end + points_per_class
     
        @staticmethod
        def show_plot():
            plt.show()

    keypoints = {0:    "Nose",1:    "Neck",2:    "RShoulder",3:    "RElbow",4:    "RWrist",5:    "LShoulder",
        6:    "LElbow",7:    "LWrist",8:    "MidHip", 9:    "RHip",10:   "RKnee",11:   "RAnkle",12:   "LHip",
        13:   "LKnee",14:   "LAnkle",15:   "REye",16:   "LEye",17:   "REar",18:   "LEar",19:   "LBigToe",
        20:   "LSmallToe",21:   "LHeel",22:   "RBigToe",23:   "RSmallToe",24:   "RHeel",}
    classes = ["boxing","drums","guitar","rowing","violin"]
    labels = [0,1,2,3,4]
    #column names for csv files
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
    files = os.listdir(r'train/')

    data = pd.read_csv(f'train/{files[1]}',names=header)
    label_column = np.ones((data.shape[0]),dtype=np.int8) *classes.index(files[0][9:-4])

    for i in range(10):
        df = pd.read_csv(f'train/{files[i]}',names=header)
        data=pd.concat([data,df])
        this_label = np.ones((df.shape[0]),dtype=np.int8) * classes.index(files[i][9:-4])
        label_column = np.concatenate((label_column,this_label))
    data.shape,label_column.shape
    data = data.fillna(0)
    #print(data)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(data)
    #print(X_reduced.shape)
    pca_vis = VisualizeScatter(fig_size=(5, 5), title='PCA 2-D Projection')
    pca_vis.add_scatters(X_reduced)
    pca_vis.show_plot()
