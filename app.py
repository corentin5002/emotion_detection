import streamlit as st
from functions import *


def main():
    import_custom_style()

    # region sidebar
    st.sidebar.title('MotusDetect Hexagone')

    # Camera selector
    camera_list = get_camera_list()

    display_camera_list = { selectbox_camera_list(camera_index) : camera_index for camera_index in camera_list}

    camera_index = None
    if camera_list:
        st.sidebar.write("Caméras connectées:")
        display_camera_index = st.sidebar.selectbox('Choix de la caméra', display_camera_list.keys())

        # camera_index = display_camera_list[display_camera_index]
    else:
        st.siderbar.write("No cameras found!")


    options = []

    # Color shifter :

    red_option = st.sidebar.slider('Rouge', 0, 100, 100)

    options.append({'name': 'red', 'value': red_option})

    red_option = st.sidebar.slider('Vert', 0, 100, 100)

    options.append({'name': 'green', 'value': red_option})

    red_option = st.sidebar.slider('Bleu', 0, 100, 100)

    options.append({'name': 'blue', 'value': red_option})

    # endregion sidebar

    # region page content
    st.title("MotusDetect - Detecteur d'émotions")

    #sub title
    st.write('Flux camera')

    if camera_index is not None:
        display_camera_flux(camera_index, options)
    # region page content

if __name__ == '__main__':
    main()