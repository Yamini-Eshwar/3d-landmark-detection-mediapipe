# --------> 3D OBJECT DETECTION FROM VEDIO<--------
import cv2
import time
import mediapipe as mp

#  Grabbing the holistic model from Mediapipe 

# Initializing the model
mp_holistic = mp.solutions.holistic

holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Initializing the drwaing utils for drawing the landmarks on image
mp_drawing = mp.solutions.drawing_utils


capture = cv2.VideoCapture(0)

# Initializing the current time and previous time for calculating FPS
previousTime = 0
currentTime = 0

while capture.isOpened():
    # capture frame by frame
    success, frame = capture.read()

    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))

    # Converting BGR TO RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    

    # Making predictions using holistic model 
    # To imprve the performance, optmizer technique to stop writting into memory and save space

    image.flags.writeable = False            # 🔒 Lock before prediction
    results = holistic_model.process(image)  # 🤖 Predict
    image.flags.writeable = True             # 🔓 Unlock before drawing

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Drawing the Facial Landmarks    info1: mp.solutions = where all ai models live hands, face, pose etc
    mp_drawing.draw_landmarks(image, 
                              results.face_landmarks,
                              mp_holistic.FACEMESH_CONTOURS, 
                              mp_drawing.DrawingSpec(
                                  color=(255, 0, 255),
                                  thickness=1,
                                  circle_radius=1 
                                  )
                              )
    
    # Drawing Right Hand Landmarks
    mp_drawing.draw_landmarks(image,
                              results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS
                              )
    
    # Drawing Right Hand Landmarks
    mp_drawing.draw_landmarks(image,
                              results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS
                              )

    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # =====Shows FPS on SCREEN====
    cv2.putText(image, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show Result
    cv2.imshow("Webcame with FPS", image)

    # Exit on ESC key
    if cv2.waitKey(5) & 0xFF == 27:
        break


capture.release()
cv2.destroyAllWindows()
    










