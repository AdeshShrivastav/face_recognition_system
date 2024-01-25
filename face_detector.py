import cv2
target = cv2.CascadeClassifier("haarcascade_profileface.xml")
vid = cv2.VideoCapture(0)
while True:
    ret,frame = vid.read()
    
    col = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = target.detectMultiScale(col,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
    cv2.imshow("face",frame)
    if cv2.waitKey(0)==ord("q"):
        break

    
vid.release()
cv2.destroyAllWindows()