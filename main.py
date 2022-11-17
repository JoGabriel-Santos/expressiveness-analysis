import cv2 as cv

from analysis import TeacherAnalysis
from analysis import StudentAnalysis

teacher_analysis = TeacherAnalysis()
student_analysis = StudentAnalysis()

teacher_video = cv.VideoCapture('util/videos/Teacher.wmv')
student_video = cv.VideoCapture('util/videos/Student.wmv')

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
    ret, frame_teacher = teacher_video.read()
    ret, frame_student = student_video.read()

    body_language_efficiency, teacher_emotion = teacher_analysis.return_teacher_analysis(frame_teacher)
    student_emotion = student_analysis.return_student_analysis(frame_student)

    print_results(teacher_emotion, student_emotion, body_language_efficiency)

    if teacher_video:
        cv.imshow('Teacher analysis', cv.resize(frame_teacher, (960, 540)))

    if student_video:
        cv.imshow('Student analysis', cv.resize(frame_student, (960, 540)))

    key = cv.waitKey(10) 

    if key == 27:
      break
