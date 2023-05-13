import tensorflow as tf
import cv2
import numpy as np
import os
import random


def write(data, img):
    text = data

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)
    thickness = 2

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    text_x = 10
    text_y = text_size[1] + 10

    cv2.putText(img, text, (text_x, text_y), font,
                font_scale, font_color, thickness)


activity_map = {0: 'Safe driving',
                1: 'Texting - right',
                2: 'Talking on the phone - right',
                3: 'Texting - left',
                4: 'Talking on the phone - left',
                5: 'Operating the radio',
                6: 'Drinking',
                7: 'Reaching behind',
                8: 'Hair and makeup',
                9: 'Talking to passenger'}

model = tf.keras.models.load_model('model/model_download.h5', compile=False)


def predict_image():
    img_list = os.listdir('data/test')
    img = os.path.join('data/test', img_list[random.randint(1, len(img_list))])
    frame = cv2.imread(img)

    data = cv2.resize(frame, (224, 224))
    data = data/255.0
    data = data.reshape(1, 224, 224, 3)
    prediction = model.predict(data)
    prediction = (activity_map[np.argmax(prediction)])
    write(prediction, frame)
    cv2.imshow('frame', frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def predict_live():
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        data = cv2.resize(frame, (224, 224))
        data = data/255.0
        data = data.reshape(1, 224, 224, 3)
        prediction = model.predict(data)
        prediction = (activity_map[np.argmax(prediction)])
        write(prediction, frame)
        cv2.imshow('frame', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


predict_image()
