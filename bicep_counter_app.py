import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import streamlit as st

st.set_page_config(page_title="AI 아령 카운터", layout="centered")
st.title("🏋️‍♂️ AI 아령 카운터")
st.markdown("실시간으로 아령 운동 횟수를 세어주는 웹앱입니다.")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

class BicepCounter(VideoProcessorBase):
    def __init__(self):
        self.counter = 0
        self.stage = None

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            angle = calculate_angle(shoulder, elbow, wrist)

            if angle > 160:
                self.stage = "down"
            if angle < 50 and self.stage == "down":
                self.stage = "up"
                self.counter += 1

            cv2.putText(image, f'Count: {self.counter}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(image, f'Angle: {int(angle)}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        except:
            pass

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return image

webrtc_streamer(key="bicep", video_processor_factory=BicepCounter)
