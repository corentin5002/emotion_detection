import cv2 as cv
import streamlit as st


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

    while True:
        # Get frames from the camera
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from the camera. Exiting...")
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        if len(options):
            apply_options(frame, options)

        # TODO: import model here

        # Display the frame
        stframe.image(frame, channels="RGB", use_container_width=True)

    # Release the video capture
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
