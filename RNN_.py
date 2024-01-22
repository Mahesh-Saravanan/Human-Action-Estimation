import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
class RNN:
    def __init__(self):
        self.normaliser = np.array(([640,480,640,480,640,480,640,480,640,480,
                    640,480,640,480,640,480,640,480,640,480,
                    640,480,640,480,640,480,640,480,640,480,
                    640,480,640,480,640,480,640,480,640,480,
                    640,480,640,480,640,480,640,480,640,480,np.pi*2,np.pi*2,np.pi*2,np.pi*2])) 
                    
    def build_model(self):
        model = keras.models.Sequential()

        #Cell 1
        model.add(LSTM(128,return_sequences = True, activation = 'relu', input_shape = (30,54)))
        model.add(layers.Dropout(0.2))

        #Cell 2
        model.add(LSTM(128,return_sequences = True, activation = 'relu'))
        model.add(layers.Dropout(0.2))

        #Cell 3
        model.add(LSTM(128,return_sequences = False, activation = 'relu'))
        model.add(layers.Dropout(0.2))



        # Fully connected layers
        model.add(layers.Dense(128,activation = "relu"))
        model.add(layers.Dense(32,activation = "relu"))
        model.add(layers.Dropout(0.2))
        #output layer
        model.add(layers.Dense(5,activation = 'softmax'))
        self.model = model
        
    def fit(self,x_train,y_train,batch = 128,lr = 0.001,ep = 100):
    
        x_train = self.norm(x_train)
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr )
        metrics = ['accuracy']
        batch_size = batch
        epochs = ep
        self.model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = metrics)
        log = self.model.fit(x_train,y_train,epochs = epochs,validation_split = 0.2,shuffle = True,batch_size = batch_size, verbose = 0)
        
    def evaluate(self, x_test,y_test):
        x_test = self.norm(x_test)
        #print(x_test)
        acc = (round(((self.model.evaluate(x_test,y_test,batch_size = 64, verbose =0))[1]*100),2))
        cm = np.zeros((5,5))
        pl = np.argmax(self.model.predict(x_test,verbose =0),axis =1)
        for i in range(len(y_test)):
            cm[y_test[i]][pl[i]]+=1
            
        return acc,cm
    def predict(self,store):
        res = []
        #print(self.norm(np.array(store[0])))
        for i in range(305):
            r = list(np.argmax(self.model.predict(self.norm(np.array(store[i])),verbose = 0),axis=1))
            res.append(max(r,key=r.count))
        return res    
    def print_summary(self):
        print(self.model.summary())
    def norm(self,data):
        #return data
        return data/self.normaliser