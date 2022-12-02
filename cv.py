from turtle import pos
import cv2
import numpy as np
from time import sleep

width_min=80 #Chiều rộng hình chữ nhật tối thiểu
height_min=80 #Chiều dài hình chữ nhật tối thiểu

offset=6 #lỗi được phép giữa các pixel 

pos_linha=550 #Đếm vị trí dòng

delay= 60 #FPS do vídeo

detec = []
count_left= 0
count_right= 0

	
def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('tracking.mp4')
subtractor = cv2.bgsegm.createBackgroundSubtractorMOG() #phép trừ nền

while True:
    ret , frame1 = cap.read()
    tempo = float(1/delay)
    sleep(tempo) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    #lọc nhiễu
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtractor.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) #chiều cao và chiều rộng là 5
    dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
    #tìm tập hợp các điểm đường viền
    contour,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255,127,0), 3) 
    cv2.line(frame1, (40, pos_linha), (560, pos_linha), (255,127,0), 3) 
    cv2.line(frame1, (700, pos_linha), (1100, pos_linha), (255,127,0), 3) 
    for(i,c) in enumerate(contour):
        #tìm bounding box và vẽ lên frame
        (x,y,w,h) = cv2.boundingRect(c) #tìm ra bouding box
        valid_contour = (w >= width_min) and (h >= height_min)
        if not valid_contour:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        centro = pega_centro(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0,255), -1)

        for (x,y) in detec:
            if y<(pos_linha+offset) and y>(pos_linha-offset) and x<575 and x>25:
                count_left+=1
                cv2.line(frame1, (40, pos_linha), (560, pos_linha), (0, 127, 255), 3) 
                detec.remove((x,y))
                print("car is detected : "+str  (count_left))  
            elif y<(pos_linha+offset) and y>(pos_linha-offset) and x>650 and x<1110:
                count_right+=1
                cv2.line(frame1, (700, pos_linha), (1100, pos_linha), (0, 127, 255), 3) 
                detec.remove((x,y))
                print("car is detected : "+str(count_right))          
       
    cv2.putText(frame1, "Vehicles left : "+str(count_left), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3)
    cv2.putText(frame1, "Vehicles right : "+str(count_right), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3)
    cv2.putText(frame1, "Total : "+str(count_right+count_left), (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3)
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detect",dilatada)

    key = cv2.waitKey(1)
    if key == ord("q"):
            break
    
cv2.destroyAllWindows()
cap.release()
