from my_CNN_model import *
import cv2
import numpy as np
import keras

import speech_recognition as sr
import threading
import os
import nltk

recognizer = sr.Recognizer()


speechChange = []
speechChange.append("red")
def task2():
    r = sr.Recognizer()
    print("thread2")
    while 1:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            print("say something ")
            audio = recognizer.listen(source)
            try:
                print("Getting tokens ")
                tokens = nltk.word_tokenize(recognizer.recognize_google(audio))
                print("Wait for tokens ")
                speechChange[0] = tokens[0]
                print(tokens)
                print("text : " + speechChange[0])
            except:
                pass



if __name__== "__main__" :

    my_model = load_my_CNN_model('my_model')
    t2=threading.Thread(target=task2, name='t2')
    t2.start()

    # Face cascade to detect faces
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

    # Define the upper and lower boundaries for a color to be considered "Blue"
    blueLower = np.array([100, 60, 60])
    blueUpper = np.array([140, 255, 255])

    # Define a 5x5 kernel for erosion and dilation
    kernel = np.ones((5, 5), np.uint8)

    # Define filters
    filters = ['images/sunglasses.png', 'images/sunglasses_4.png', 'images/sunglasses_5.jpg', 'images/sunglasses_6.png']
    sunglasses_Filter = {"black" : 0, "green" : 1, "red" : 2, "gray" : 3, "pink" : 2, "blue": 3 }

    filterIndex = 0

    # Hat images
    hat_images = ['images/hat_1.png', 'images/hat_2.png', 'images/hat_3.png']
    # Change this to change hat in frame
    hat_Filter = {"black" : 0, "green" : 1, "red" : 2, "gray" : 0, "pink" : 1, "blue": 2}

    hatIndex = 2
    # Mustache images
    must_images = ['images/mustache_1.png', 'images/mustache_2.png', 'images/mustache_3.png']
    must_Filter = {"black" : 0, "green" : 1, "red" : 2, "gray" : 0, "pink" : 1, "blue": 2}
    # Change this to change mustache in frame
    mustIndex = 2

    RGBValue = {"green" : (0,255,0), "red" : (255,0,0), "blue" : (0,0,255), "black" : (0,0,0), "gray" : (128,128,128), "pink" : (255,192,203)}
    DefaultValue = (0,255,0)

    filterIndex_moustache  = 0
    camera = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('FinalVideo.avi',fourcc, 20.0, (640,480))
    t2=threading.Thread(target=task2, name='t2')
   # t1.start()
    t2.start()
    print("mehul be bi=ola hai")
        # Keep looping
    while True:
        # Grab the current paintWindow
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        frame2 = np.copy(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		
        # changing the sunglasses, Hat & Mustache with speech
        if speechChange[0] in sunglasses_Filter:
            filterIndex = sunglasses_Filter[speechChange[0]]
            hatIndex = hat_Filter[speechChange[0]]
            mustIndex = must_Filter[speechChange[0]]
            DefaultValue = RGBValue[speechChange[0]]

        # Add the 'Filter' To Show The color change of the filter objects on the face
        frame = cv2.rectangle(frame, (500,10), (620,65), (0,0,0), -1)
        cv2.putText(frame, "FILTER", (508, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, DefaultValue, 2, cv2.LINE_AA)
  #      print('checking 2')
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.25, 6)

#        print('checking 4')
        for (x, y, w, h) in faces:

            # Grab the face
            gray_face = gray[y:y+h, x:x+w]
            color_face = frame[y:y+h, x:x+w]

            # Normalize to match the input format of the model - Range of pixel to [0, 1]
            gray_normalized = gray_face / 255

            # Resize it to 96x96 to match the input format of the model
            original_shape = gray_face.shape # A Copy for future reference
            face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
            face_resized_copy = face_resized.copy()
            face_resized = face_resized.reshape(1, 96, 96, 1)

            # Predicting the keypoints using the model
            keypoints = my_model.predict(face_resized)

            # De-Normalize the keypoints values
            keypoints = keypoints * 48 + 48

            # Map the Keypoints back to the original image
            face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)
            face_resized_color2 = np.copy(face_resized_color)

            # Pair them together
            points = []
            for i, co in enumerate(keypoints[0][0::2]):
                points.append((co, keypoints[0][1::2][i]))



            # Add Sunglasses FILTER to the frame
            
            sunglasses = cv2.imread(filters[filterIndex], cv2.IMREAD_UNCHANGED)
            sunglass_width = int((points[7][0]-points[9][0])*1.1)
            sunglass_height = int((points[10][1]-points[8][1])/1.1)
            sunglass_resized = cv2.resize(sunglasses, (sunglass_width, sunglass_height), interpolation = cv2.INTER_CUBIC)
            transparent_region = sunglass_resized[:,:,:3] != 0
            face_resized_color[int(points[9][1]):int(points[9][1])+sunglass_height, int(points[9][0]):int(points[9][0])+sunglass_width,:][transparent_region] = sunglass_resized[:,:,:3][transparent_region]
            
             # mimic sunglasses lines for mustache
            mustache = cv2.imread(must_images[mustIndex], cv2.IMREAD_UNCHANGED)
            # (width, height) for each mustache
            must_sizes = [[int((points[11][0] - points[12][0])*1.6), int(points[13][1] - points[10][1])], [int((points[11][0] - points[12][0])*1.6), int((points[13][1] - points[10][1])*2.0)], [int((points[11][0] - points[12][0])*2.3), int((points[13][1] - points[10][1])*2.0)]]
            # (offset_y, offset_x) for each mustache
            offsets = [[int(0.2*must_sizes[mustIndex][1]), int(-0.2*must_sizes[mustIndex][0])], [int(0.1*must_sizes[mustIndex][1]), int(-0.2*must_sizes[mustIndex][0])], [int(0.05*must_sizes[mustIndex][1]), int(-0.28*must_sizes[mustIndex][0])]]
            must_resized = cv2.resize(mustache, (must_sizes[mustIndex][0], must_sizes[mustIndex][1]), interpolation=cv2.INTER_CUBIC)
            if mustIndex == 0:
                transparent_region = must_resized[:, :, :3] == 0
            else:
                transparent_region = must_resized[:,:,:3] != 0
            face_resized_color[offsets[mustIndex][0]+int(points[10][1]):offsets[mustIndex][0]+int(points[10][1])+must_sizes[mustIndex][1],offsets[mustIndex][1]+int(points[12][0]):offsets[mustIndex][1]+int(points[12][0])+must_sizes[mustIndex][0],:][transparent_region] = must_resized[:,:,:3][transparent_region]

            # mimic sunglasses lines for hat
            # load hat image
            hat = cv2.imread(hat_images[hatIndex], cv2.IMREAD_UNCHANGED)
            # (width, height) for each hat
            hat_sizes = [[int(1.3*w), int(1.1*h)], [int(1.12*w), int(1.0*h)], [int(1.5*w), int(1.35*h)]]
            hat_resized = cv2.resize(hat, (hat_sizes[hatIndex][0], hat_sizes[hatIndex][1]), interpolation=cv2.INTER_CUBIC)
            transparent_region = hat_resized[:, :, :3] != 0

            # Resize the face_resized_color image back to its original shape
            frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation=cv2.INTER_CUBIC)
            # (offset_y, offset_x) for each hat
            offsets = [[int(-0.65*h), int(-0.17*w)], [int(-0.55*h), int(-0.06*w)], [int(-0.7*h), int(-0.1*w)]]
            frame[offsets[hatIndex][0]+y:offsets[hatIndex][0]+y+hat_sizes[hatIndex][1],offsets[hatIndex][1]+x:offsets[hatIndex][1]+x+hat_sizes[hatIndex][0],:][transparent_region] = hat_resized[:, :, :3][transparent_region]


            # Resize the face_resized_color image back to its original shape
            #frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)

            # Add KEYPOINTS to the frame2
            for keypoint in points:
                cv2.circle(face_resized_color2, keypoint, 1, (0,255,0), 1)

            frame2[y:y+h, x:x+w] = cv2.resize(face_resized_color2, original_shape, interpolation = cv2.INTER_CUBIC)

            # Show the frame and the frame2
            cv2.imshow("Selfie Filters", frame)
            out.write(frame)
    #        cv2.imshow("Facial Keypoints", frame2)

        # If the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

   #     print('checking 6')
    # Cleanup the camera and close any open windows
    camera.release()
    out.release()
    cv2.destroyAllWindows()
   # t1=threading.Thread(target=task1, name='t1')

   # t1.join()
    t2.join()