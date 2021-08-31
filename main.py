import cv2
from flask import Flask, current_app
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import base64
import face_recognition
import numpy as np
import json

import requests

is_front_end_open = False

app = Flask(__name__)

app_ctx = app.app_context()
app_ctx.push()

CORS(app)

app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

video_capture = cv2.VideoCapture(0)
known_face_encodings = []
known_face_names = []


# @socketio.on_error():
#     print(e)

def start_socket_server():
     socketio.run(app)

def get_faces():
    url = 'http://localhost:3000/api/faces/getFaces'
    headers = {'Content-type': 'application/json'}
    response = requests.get(url,{} , headers=headers)
    known_face_encodings.clear()
    for face in response.json():
        known_face_encodings.append(np.asarray(face['encoded_face']))


def save_face(encoded_face, rgb_small_frame):
    url = 'http://localhost:3000/api/faces/save_face'
    retval, buffer = cv2.imencode('.jpg', rgb_small_frame)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    response = requests.post(url, {"encoded_face": json.dumps(encoded_face.tolist()), "image":jpg_as_text})

@socketio.on('connect')
def test_connect():
    get_faces()
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('start_streaming')
def emit_start_streaming():
    # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH ,300)
    # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
    while (True):
        # Capture the video frame
        # by frame
        ret, frame = video_capture.read()
        start_face_detection_recognition(frame)
        retval, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        emit('frame', jpg_as_text)

def start_face_detection_recognition(frame):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)

    if len(face_locations) != 0:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
             #Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            # If a match was found in known_face_encodings, just use the first one.
            if len(face_distances) != 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    print('we have that face')
                else:
                    save_face(face_encoding, rgb_small_frame)
                    get_faces()
            else:
                save_face(face_encoding, rgb_small_frame)
                get_faces()


if __name__ == '__main__':
    start_socket_server()


