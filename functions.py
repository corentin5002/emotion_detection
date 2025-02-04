import cv2 as cv
import streamlit as st
import mediapipe as mp
import tensorflow as tf
import numpy as np
import onnxruntime as ort
import time

# Liste des émotions (à adapter selon votre dataset)
emotions = ["Colere", "Degout", "Peur", "Bonheur", "Neutre", "Tristesse", "Surprise"]
emotions_onnx = ["Colere", "Mepris", "Degout", "Peur", "Bonheur", "Neutre", "Tristesse", "Surprise"]

def get_model_tf():
    # Charger le modèle de prédiction d'émotion
    model = tf.keras.models.load_model("face_modele.h5")
    return model

def get_model_onnx():
    session = ort.InferenceSession("resnet18.onnx")
    return session

def get_camera_list():
    available_cameras = []
    for index in range(5):
        cap = cv.VideoCapture(index)

        if cap.isOpened():
            available_cameras.append(index)
            cap.release()

    return available_cameras


def selectbox_camera_list(camera_index):
    return f"Camera {camera_index}" if camera_index > 0 else f'Camera {camera_index}  (par defaut)'


def display_camera_flux(camera, options=None):
    cap = cv.VideoCapture(camera)

    stframe = st.empty()  # Placeholder for the video frames
    st.write("Press 'Stop' in the Streamlit menu to stop the feed.")

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    model = get_model_tf()
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:

        while True:
            # Get frames from the camera
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from the camera. Exiting...")
                break

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), 
                                  int(bboxC.width * iw), int(bboxC.height * ih))

                    face = frame[y:y+h, x:x+w]  # Extraire le visage

                    if face.size > 0:  # Vérifier si un visage a bien été extrait
                        face_input = preprocess_face(face)
                        predictions = model.predict(face_input)
                        emotion_label = emotions[np.argmax(predictions)]

                        # Affichage de l'émotion détectée
                        cv.putText(frame, emotion_label, (x, y-10), 
                                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Dessiner le rectangle autour du visage
                    mp_drawing.draw_detection(frame, detection)

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Display the frame
            stframe.image(frame, channels="RGB", use_container_width=True)

            time.sleep(0.1)  # Pause de 0.1s pour éviter la surcharge du CPU

        # Release the video capture
        cap.release()

def display_camera_flux_onnx(camera, options=None):
    cap = cv.VideoCapture(camera)

    stframe = st.empty()  # Placeholder pour le flux vidéo
    st.write("Press 'Stop' in the Streamlit menu to stop the feed.")

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    model_onnx = get_model_onnx()

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from the camera. Exiting...")
                break

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), 
                                  int(bboxC.width * iw), int(bboxC.height * ih))

                    face = frame[y:y+h, x:x+w]  # Extraire le visage

                    if face.size > 0:  # Vérifier si un visage a bien été extrait
                        face_input = preprocess_face_onnx(face)  # Prétraitement
                        emotion_label = predict_emotion_onnx(model_onnx, face_input)  # Prédiction

                        # Affichage de l'émotion détectée
                        cv.putText(frame, emotion_label, (x, y-10), 
                                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Dessiner le rectangle autour du visage
                    mp_drawing.draw_detection(frame, detection)

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB", use_container_width=True)

            time.sleep(0.1)  # Pause de 0.1s pour éviter la surcharge du CPU

        cap.release()


def apply_options(frame, options):
    colors = ['red', 'green', 'blue']

    for option in options:
        if option['name'] in colors:
            color_canal = colors.index(option['name'])
            shift_color(frame, color_canal, option['value'])


def shift_color(frame, color_canal, value):
    # shift color_canal channel applying value in value (0 to 100)
    channel = frame[:, :, color_canal].astype('float32')
    channel *= value / 100
    channel = channel.astype('uint8')
    frame[:, :, color_canal] = channel

def preprocess_face(face_image):
    """Prépare l'image du visage pour la prédiction du modèle."""
    target_size = (48, 48)
    face_resized = cv.resize(face_image, target_size)
    face_gray = cv.cvtColor(face_resized, cv.COLOR_BGR2GRAY)
    face_normalized = face_gray / 255.0  # Normalisation
    face_input = np.expand_dims(face_normalized, axis=[0, -1])  # Ajout des dimensions batch et channel
    return face_input

def preprocess_face_onnx(face_image):
    """Prépare l'image du visage pour la prédiction du modèle ONNX."""
    target_size = (224, 224)  # Adapter à la taille attendue par le modèle
    face_resized = cv.resize(face_image, target_size)
    face_rgb = cv.cvtColor(face_resized, cv.COLOR_BGR2RGB)  # Garder les 3 canaux
    face_normalized = face_rgb / 255.0  # Normalisation entre 0 et 1
    face_transposed = np.transpose(face_normalized, (2, 0, 1))  # Changer de NHWC -> NCHW
    face_input = np.expand_dims(face_transposed, axis=0).astype(np.float32)  # Ajouter batch + float32
    return face_input

def predict_emotion_onnx(model_session, face_input):
    """Effectue une prédiction d'émotion avec le modèle ONNX."""
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name
    
    prediction = model_session.run([output_name], {input_name: face_input})[0]

    if prediction is None or len(prediction) == 0:  # Vérifier si la sortie est vide
        return ""

    emotion_index = np.argmax(prediction)
    
    if emotion_index >= len(emotions_onnx):  # Vérifier si l'index est valide
        return "Inconnue"

    return emotions_onnx[emotion_index]
