import matplotlib.pyplot as plt
import numpy as np
def compare(cnn_1d_acc, cnn_2d_acc,rnn_acc,rf_acc,cnn_1d_cm, cnn_2d_cm,rnn_cm,rf_cm,classes):   
    fig, (ax1,ax2,ax3,ax4)= plt.subplots(1,4,figsize=(25, 25))
    fs = 20
    ax1.matshow(cnn_1d_cm)
    ax1.set_title('CNN 1d', size = fs)
    ax1.set_xlabel ('Predictions')
    ax1.set_ylabel ('True Classes')
    ax1.set_xticks(range(5),classes)
    ax1.set_yticks(range(5),classes)
    for (i, j), z in np.ndenumerate(cnn_1d_cm):
        ax1.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    ax2.matshow(cnn_2d_cm)
    ax2.set_title('CNN 2d', size = fs)
    ax2.set_xlabel ('Predictions')
    ax2.set_ylabel ('True Classes')
    ax2.set_xticks(range(5),classes)
    ax2.set_yticks(range(5),classes)    
    for (i, j), z in np.ndenumerate(cnn_2d_cm):
        ax2.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    ax3.matshow(rnn_cm)
    ax3.set_title('LSTM', size = fs)
    ax3.set_xlabel ('Predictions')
    ax3.set_ylabel ('True Classes')
    ax3.set_xticks(range(5),classes)
    ax3.set_yticks(range(5),classes)    
    for (i, j), z in np.ndenumerate(rnn_cm):
        ax3.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    ax4.matshow(rf_cm)
    ax4.set_title('Random forest', size = fs)
    ax4.set_xlabel ('Predictions')
    ax4.set_ylabel ('True Classes')
    ax4.set_xticks(range(5),classes)
    ax4.set_yticks(range(5),classes)    
    for (i, j), z in np.ndenumerate(rf_cm):
        ax4.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    plt.show()
    print(f"\t     {cnn_1d_acc}%\t\t\t      {cnn_2d_acc}%\t\t\t      {rnn_acc}%\t\t\t      {rf_acc}%")
    
def dist (res_cnn_1d,res_cnn_2d,res_rnn,res_rf,classes):
    plt.figure(figsize = (9,5))
    plt.subplot(2,2,1)
    plt.hist(res_cnn_1d,bins = 5)
    plt.title('CNN 1D')
    plt.xticks(range(5),classes)
    plt.subplot(2,2,2)
    plt.hist(res_cnn_2d,bins = 5)
    plt.title('CNN 2D')
    plt.xticks(range(5),classes)
    plt.subplot(2,2,3)
    plt.hist(res_rnn,bins = 5)
    plt.title('LSTM')
    plt.xticks(range(5),classes)
    plt.subplot(2,2,4)
    plt.hist(res_rf,bins = 5)
    plt.title('Random forest')
    plt.xticks(range(5),classes)
    plt.tight_layout(pad = 1,w_pad = 10, h_pad = 1.0)
    plt.show()
    