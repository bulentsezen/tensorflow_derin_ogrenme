import cv2
import dlib
import numpy as np
import time
from keras.models import load_model
from scipy.spatial import distance
from imutils import face_utils

predictor = dlib.shape_predictor("/home/isuzu/Downloads/DrowsinessDetect/shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('/home/isuzu/Downloads/DrowsinessDetect/haarcascade_frontalface_alt.xml')


def detect(img, cascade = face_cascade , minimumFeatureSize=(20, 20)):
    if cascade.empty():
        raise (Exception("There was a problem loading your Haar Cascade xml file."))
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)
    
  
    if len(rects) == 0:
        return []

    rects[:, 2:] += rects[:, :2]

    return rects
def rect_to_bb(rect):
	
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 

	return (x, y, w, h)
def cropEyes(frame):
     
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    te = detect(gray, minimumFeatureSize=(80, 80))


    if len(te) == 0:
        return None
    elif len(te) > 1:
        face = te[0]
    elif len(te) == 1:
        [face] = te


    face_rect = dlib.rectangle(left = int(face[0]), top = int(face[1]),
                                right = int(face[2]), bottom = int(face[3]))
    
  
    shape = predictor(gray, face_rect)
    shape = face_utils.shape_to_np(shape)
    
    (x, y, w, h) = face_utils.rect_to_bb(face_rect)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.putText(frame, "Face #{}".format(1), (x-10, y-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    for (x, y) in shape:
        cv2.circle(frame, (x,y), 1,(0, 0, 255), -1)



    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

 
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    global ear
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0

    
    
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
    

    l_uppery = min(leftEye[1:3,1])
    l_lowy = max(leftEye[4:,1])
    l_dify = abs(l_uppery - l_lowy)

    lw = (leftEye[3][0] - leftEye[0][0])

 
    minxl = (leftEye[0][0] - ((34-lw)/2))
    maxxl = (leftEye[3][0] + ((34-lw)/2)) 
    minyl = (l_uppery - ((26-l_dify)/2))
    maxyl = (l_lowy + ((26-l_dify)/2))
    

    left_eye_rect = np.rint([minxl, minyl, maxxl, maxyl])
    left_eye_rect = left_eye_rect.astype(int)
    left_eye_image = gray[(left_eye_rect[1]):left_eye_rect[3], (left_eye_rect[0]):left_eye_rect[2]]
    

    r_uppery = min(rightEye[1:3,1])
    r_lowy = max(rightEye[4:,1])
    r_dify = abs(r_uppery - r_lowy)
    rw = (rightEye[3][0] - rightEye[0][0])
    minxr = (rightEye[0][0]-((34-rw)/2))
    maxxr = (rightEye[3][0] + ((34-rw)/2))
    minyr = (r_uppery - ((26-r_dify)/2))
    maxyr = (r_lowy + ((26-r_dify)/2))
    right_eye_rect = np.rint([minxr, minyr, maxxr, maxyr])
    right_eye_rect = right_eye_rect.astype(int)
    right_eye_image = gray[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]

  
    if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
        return None
  
    left_eye_image = cv2.resize(left_eye_image, (34, 26))
    right_eye_image = cv2.resize(right_eye_image, (34, 26))
    right_eye_image = cv2.flip(right_eye_image, 1)
  
    return left_eye_image, right_eye_image 


def cnnPreprocess(img):
    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img, axis=2)
    img = np.expand_dims(img, axis=0)
    return img

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
   
def main():
    
 
    camera = cv2.VideoCapture("/home/isuzu/Downloads/DrowsinessDetect/video-1527941609.mp4")
    model = load_model('/home/isuzu/Downloads/DrowsinessDetect/blinkModel.hdf5')
    
  
    close_counter = blinks = mem_counter= 0
    state = ''
    durum = ''
    
    start_time = time.time()
    x = 1 
    counter = 0
    counter1 = 0
    frame_counter=48
    counter2 = 0
    frame_counter2=48
    
    while True:
        

        ret, frame = camera.read()
       
       
        eyes = cropEyes(frame)
        if eyes is None:
            continue
        else:
            left_eye,right_eye = eyes


        
        prediction = (model.predict(cnnPreprocess(left_eye)) + model.predict(cnnPreprocess(right_eye)))/2.0
            

        if prediction > 0.2 :
            state = 'open'
            close_counter = 0
        else:
            state = 'close'
            close_counter += 1
        

        if state == 'open' and mem_counter > 1:
            blinks += 1
    
        mem_counter = close_counter 
        
        if ear < 0.20 and close_counter >=15 :
            counter1+=1
            if counter1 >=frame_counter :
                counter2 = 0
                durum = 'uyuyor'
        elif ear >= 0.20 and ear <= 0.30 :
            counter2+=1
            if counter2 >=frame_counter2 :
                counter1 = 0
                durum = 'uyku geldi'
            
            
        else:
            counter1 = 0
            counter2 = 0
            durum = 'uyanik'
            

     
 

        cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Eyes: {}".format(state), (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Count: {}".format(counter1), (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)    
        cv2.putText(frame, "Durum: {}".format(durum), (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        counter+=1
        if (time.time() - start_time) > x :
            print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()   
        
      
        cv2.imshow('Drowsiness Detect', frame)
        key = cv2.waitKey(1) & 0xFF
        
    
        if key == ord('q'):
            break
   
 
    cv2.destroyAllWindows()
    del(camera)


if __name__ == '__main__':
    main()
