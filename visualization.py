import cv2
import numpy as np
import time
def visualize(data,label,classes):
    colors = [(0,20,155),(0,40,155),(0,60,155),(0,80,155),(0,100,155),
          (100,120,255),(100,140,255),(0,180,155),(0,200,155),(0,220,155),
          (100,240,255),(100,240,255),(40,240,255),(60,240,155),(80,240,155),
         (100,240,255),(20,240,255),(40,240,255),(160,240,255),(180,240,255),
         (200,240,255),(220,240,255),(240,240,255),(255,240,255),(255,240,0),
             (255,0,240),(255,30,240),(255,80,240)]
    
    for j in range(data.shape[0]):
        img = np.zeros((640,480,3),dtype=np.uint8)
        counter=0

        for i in range(0,54,2):


            y = int(data[j,i])
            x = int(data[j,i+1])

            img = cv2.circle(img, (y,x), 2, colors[counter], thickness=10, lineType=8, shift=0) 
            counter=counter+1

        img = cv2.putText(img,classes[label],(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
        
        cv2.imshow(f'im',img)
        time.sleep(0.0005)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()
    
