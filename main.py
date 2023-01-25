import numpy as np
import cv2 as cv

from analysis import TeacherAnalysis
from analysis import StudentAnalysis

teacher_analysis = TeacherAnalysis()
student_analysis = StudentAnalysis()

teacher_video = cv.VideoCapture('util/videos/Teacher.wmv')
student_video = cv.VideoCapture('util/videos/Student.wmv')

video = cv.VideoCapture('util/videos/Video.mp4')

analyzed_frames = 0
matching_emotions = 0


def print_results(teacher_emotion, student_emotion, body_language_efficiency):
    global analyzed_frames, matching_emotions

    analyzed_frames += 1

    if teacher_emotion == student_emotion:
        matching_emotions += 1

    print('Teacher emotion: ' + teacher_emotion)
    print('Student emotion: ' + student_emotion)
    print('Matching emotions: ' + str(round(matching_emotions * 100 / analyzed_frames, 1)) + '%')

    print('Body language efficiency: ' + body_language_efficiency + '%')


while True:
    ret, frame = video.read()

    body_language_efficiency, teacher_emotion_and_coordinates = teacher_analysis.return_teacher_analysis(frame[482:858, 595:1380])
    student_emotion = student_analysis.return_student_analysis(frame[0:475, 0:1914])

    # Teacher analysis
    teacher_emotion = str(teacher_emotion_and_coordinates[0][0])
    teacher_facial_coordinates = teacher_emotion_and_coordinates[1]

    x = teacher_facial_coordinates[0] + 595
    y = teacher_facial_coordinates[1] + 482
    w = teacher_facial_coordinates[2]
    h = teacher_facial_coordinates[3]

    cv.putText(frame, teacher_emotion, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cv.rectangle(frame, (595, 1380), (482, 858), (255, 0, 0), 2)

    print_results(teacher_emotion, student_emotion, body_language_efficiency)

    cv.imshow('Analysis', cv.resize(frame, (960, 540)))

    key = cv.waitKey(100)

    if key == 27:
        break

cv.destroyAllWindows()
