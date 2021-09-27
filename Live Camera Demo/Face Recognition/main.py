import cv2
import face_recognition
import numpy as np



# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

sel_image=face_recognition.load_image_file('C:\proj\Face Recognition\Face Recognition\known image\Selena gomez.jpg')
jim_image=face_recognition.load_image_file('C:\proj\Face Recognition\Face Recognition\known image\Jim-Carreyjpg.jpg')
obama_image=face_recognition.load_image_file('C:\proj\Face Recognition\Face Recognition\known image\Barack-Obama.jpg')
dicaprio_image=face_recognition.load_image_file('C:\proj\Face Recognition\Face Recognition\known image\dicapriojpg.jpg')
tobey_image=face_recognition.load_image_file('C:\proj\Face Recognition\Face Recognition\known image\Tobey_Maguire.jpg')
barath_image=face_recognition.load_image_file('C:\proj\Face Recognition\Face Recognition\known image\Baarath.jpg')


sel_encoding=face_recognition.face_encodings(sel_image)[0]
jim_encoding=face_recognition.face_encodings(jim_image)[0]
obama_encoding=face_recognition.face_encodings(obama_image)[0]
dicaprio_encoding=face_recognition.face_encodings(dicaprio_image)[0]
tobey_encoding=face_recognition.face_encodings(tobey_image)[0]
barath_encoding=face_recognition.face_encodings(barath_image)[0]

known_face_names=['Selena Gomez','Jim Carrey','Obama','Leonardo Dicaprio','Tobey Maguire','Baarathan']
known_face_encodings=[sel_encoding,jim_encoding,obama_encoding,dicaprio_encoding,tobey_encoding,barath_encoding]







while True:
    

    ret, frame = video_capture.read()

    # Resize frame of video to 1/5 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.20, fy=0.20)
    
    face_locations=face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)


    face_names=[]
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            face_names.append(name)



    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # Scale back up face locations since the frame we detected in was scaled to 1/5 size
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5


        #Drawing Rectangle cover the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #Drawing Filled rectangle background for text
        cv2.rectangle(frame, (left, bottom), (right, bottom-20), (255, 0, 255), cv2.FILLED)


    #Put Text method is used to put text on image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)



#For converting BGR to RGB
# 

# unknown_image_rgb = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)


    cv2.imshow("Video",frame)


# Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()    