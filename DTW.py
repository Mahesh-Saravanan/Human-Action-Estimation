import numpy as np 
import matplotlib.pyplot as plt
class DynamicTimeWraping:
    def fit (self,tf,tl,classes):
        self.tf = tf
        self.tl = tl
        self.classes = classes
        
    def predict(self,file):
        sample_files = self.tf
        class_scores =[]
        print("Predicting...")
        for i in range(5):
            
            Rp = int(((i+1)/5)*100)
            gap = ' '
            sign = '='    
            print('\r','[',(sign*Rp)+str('>')+gap*(100-Rp),']',"{} %".format(Rp),end = "")
            class_scores.append(self.alignment_score(file,sample_files[i]))
        res = np.argmin(class_scores) 
        return self.tl[res],self.classes[res]
    def alignment_score(self,file1,file2):
        cost = []
        for d in range(54):
            seq1 =file1.T[d] 
            seq2 = file2.T[d]
            N = seq1.shape[0]
            M = seq2.shape[0]
            dist_mat = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    dist_mat[i, j] = abs(seq1[i] - seq2[j])

            N, M = dist_mat.shape
            cost_mat = np.zeros((N + 1, M + 1))
            for i in range(1, N + 1):cost_mat[i, 0] = np.inf
            for i in range(1, M + 1):cost_mat[0, i] = np.inf
            for i in range(N):
                for j in range(M):
                    add_factor = min([cost_mat[i, j], cost_mat[i, j + 1], cost_mat[i + 1, j]])

                    cost_mat[i + 1, j + 1] = dist_mat[i, j] + add_factor     
            cost_mat = cost_mat[1:, 1:]        
            cost.append(cost_mat[N-1,M-1]/(N+M))
        return np.mean(cost)
    
    def plot(self,f1,f2,part,header,classes,dtwfiles,dtwlabels):
        x = dtwfiles[f1].T[part]
        y = dtwfiles[f2].T[part]
        fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
        fig.set_size_inches(18.5, 10.5)
        fig.text(0.04, 0.5, 'Pixels', va='center', rotation='vertical')
        fig.text(0.5, 0.04, 'Frames', ha='center')
        fig.suptitle(f'{header[part]} movement of {classes[dtwlabels[f1]]} and {classes[dtwlabels[f2]]} files  ', fontsize=20)
        ax1.plot(x,label = classes[dtwlabels[f1]])
        ax1.legend(loc="upper right",fontsize=20)
        ax1.set_xlim([0, 700])
        ax2.plot(y, label = classes[dtwlabels[f2]])
        ax2.legend(loc="upper right",fontsize=20)
        ax2.set_xlim([0, 700])
        plt.show()