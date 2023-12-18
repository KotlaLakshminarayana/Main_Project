from flask import Flask, jsonify, render_template, request, redirect, url_for, Response
import joblib
import numpy as np
import cv2
import librosa
from keras.models import load_model, model_from_json

app = Flask(__name__, static_folder='static')

# Load models
text_emotion_model = joblib.load('models/text_emotion_model.pkl')
audio_emotion_model = load_model('models/audio_emotion_model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('models/Emotion_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("models/Emotion_Model.h5")
print("Loaded model from disk")

# Define emotion labels
emotion_labels = ['Angry', 'Happy', 'Fear', 'Disgust', 'Sad', 'Surprise', 'Neutral']

# Initialize variables for saving video
video_writer = None
video_frames = []

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/select_model')
def select_model():
    return render_template('select_model.html')

@app.route('/input_text', methods=['GET', 'POST'])
def input_text():
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        predicted_emotion = text_emotion_model.predict([input_text])[0]
        return render_template('output.html', emotion=predicted_emotion)
    return render_template('input_text.html')

@app.route('/input_audio', methods=['GET', 'POST'])
def input_audio():
    if request.method == 'POST':
        audio_file = request.files['audio']
        if audio_file:
            # Save the uploaded audio temporarily
            audio_path = 'temp_audio.wav'
            audio_file.save(audio_path)

            # Load and preprocess the audio
            audio, sr = librosa.load(audio_path, sr=None, duration=4.0)  # Adjust duration as needed

            # Pad or truncate the audio to the desired length (192000 samples)
            desired_length = 192000
            if len(audio) < desired_length:
                silence = np.zeros(desired_length - len(audio))
                audio = np.concatenate((audio, silence))
            else:
                audio = audio[:desired_length]

            # Normalize the audio data
            audio = audio / np.max(np.abs(audio))

            # Extract MFCC features
            mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # Adjust n_mfcc as needed

            # Resize the MFCC features to match the model's input shape
            desired_shape = (400, 480)
            resized_features = cv2.resize(mfcc_features, (desired_shape[1], desired_shape[0]))

            # Add a channel dimension to match the model's input shape
            reshaped_features = resized_features.reshape(1, desired_shape[0], desired_shape[1], 1)

            # Make predictions using the model
            predicted_emotion_probabilities = audio_emotion_model.predict(reshaped_features)
            predicted_emotion_index = np.argmax(predicted_emotion_probabilities)

            # Map the index to the corresponding emotion label using the updated emotion labels
            emotion_labels = {
                1: 'Neutral',
                2: 'Calm',
                3: 'Happy',
                4: 'Sad',
                5: 'Angry',
                6: 'Fearful',
                7: 'Disgust',
                8: 'Surprised'
            }
            predicted_emotion = emotion_labels[predicted_emotion_index]

            # Cleanup: remove the temporary audio file
            import os
            os.remove(audio_path)

            return render_template('output.html', emotion=predicted_emotion)
    return render_template('input_audio.html')

@app.route('/input_video', methods=['GET','POST'])
def video_emotion():
    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render_template('select_model.html')

if __name__ == '__main__':
    app.run(debug=True)
