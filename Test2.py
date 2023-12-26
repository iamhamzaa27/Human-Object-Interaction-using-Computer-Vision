import cv2
import numpy as np
from tensorflow import keras

# Load your Keras model
model_path = "D:/Semester 7/ML/Project/HOI_model_416(1).h5"
model = keras.models.load_model(model_path)

# Video input
video_path = "D:\Semester 7\ML\Project/false4.mp4"
cap = cv2.VideoCapture(video_path)

# Frame extraction parameters
frames_per_second = 1

while True:
    success, frame = cap.read()

    if not success:
        break

    # Resize the frame to the input size of your model (416x416)
    resized_frame = cv2.resize(frame, (416, 416))

    # Preprocess the frame for your model
    test_input = resized_frame.reshape((1, 416, 416, 3)) / 255.0  # Assuming your model expects values in [0, 1]

    # Make prediction
    prediction = model.predict(test_input)
    result = round(prediction[0][0])

    # Display frames where interaction is detected
    if result == 1:
        current_second = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FPS))
        cv2.imshow(f"Interaction Detected at {current_second} seconds", resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
