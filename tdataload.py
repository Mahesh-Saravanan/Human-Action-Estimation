import pandas
import numpy as np

def test_dataload(hp,header,folder,tfiles):
    store=[]
    print("Fetching Evaluation files")
    for j in range(tfiles):
        data = pandas.read_csv(f'{folder}{j}.csv',names=header)
        data = data.drop(data.columns[[2, 5, 8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,59,62,65,68,71,74]], axis=1)
        data = data.to_numpy()
        data = np.nan_to_num(data)
        test = []
        for i in range(int(data.shape[0]/hp)):
            test.append(data[i*hp:(i*hp)+hp,:])
        store.append(test)
        Rp = int(((j+1)/tfiles)*100)
        gap = ' '
        sign = '='    
        print('\r','[',(sign*Rp)+str('>')+gap*(100-Rp),']',"{} %".format(Rp),end = "")    
    print("\n Evaluation files loaded")
    return store
        
