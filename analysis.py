import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import cv2 as cv

from keras.models import load_model, model_from_json


class BodyLanguageAnalysis:

    MEDIA_POSE = mp.solutions.pose.Pose(min_detection_confidence=0, min_tracking_confidence=0)

    analyzed_frames = 0
    freq_handmov = 0

    def predict_pose_landmarks(self, frame):

        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_HAND = 21
        RIGHT_HAND = 22

        BodyLanguageAnalysis.analyzed_frames += 1

        gray_fr = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray_fr.flags.writeable = False

        results = BodyLanguageAnalysis.MEDIA_POSE.process(gray_fr)

        gray_fr.flags.writeable = True
        gray_fr = cv.cvtColor(gray_fr, cv.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            if (landmarks[LEFT_HAND].y <= landmarks[LEFT_ELBOW].y) or (
                    landmarks[RIGHT_HAND].y <= landmarks[RIGHT_ELBOW].y):

                BodyLanguageAnalysis.freq_handmov += 1

        except:
            pass

        return str(round(BodyLanguageAnalysis.freq_handmov * 100 / BodyLanguageAnalysis.analyzed_frames, 1))


class FacialExpressionAnalysis(object):

    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model_json_file, model_weights_file):

        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_emotion(self, frame):

        global roi

        facec = cv.CascadeClassifier('util/haarcascade/haarcascade_frontalface_default.xml')

        gray_fr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]

            roi = cv.resize(fc, (48, 48))

        self.preds = self.loaded_model.predict(roi[np.newaxis, :, :, np.newaxis])

        return FacialExpressionAnalysis.EMOTIONS_LIST[np.argmax(self.preds)]


body_language = BodyLanguageAnalysis()
facial_expression = FacialExpressionAnalysis('util/model/model.json', 'util/model/model.h5')


class TeacherAnalysis:

    def return_teacher_analysis(self, frame):

        return body_language.predict_pose_landmarks(frame), facial_expression.predict_emotion(frame)


class StudentAnalysis:

    def return_student_analysis(self, frame):

        facial_expressions = []
        facial_expressions.append(facial_expression.predict_emotion(frame))

        try:
            return (max(set(facial_expressions), key=facial_expressions.count))

        except:
            pass
