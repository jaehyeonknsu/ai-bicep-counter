import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from streamlit_pose import PoseDetector
import cv2
import numpy as np

st.set_page_config(page_title="AI 아령 카운터", layout="centered")
st.title("🏋️‍♂️ AI 아령 카운터")
st.markdown("실시간으로 아령 운동 횟수를 세어주는 웹앱입니다.")

# 각도 계산 함수
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

class BicepCounter(VideoProcessorBase):
    def __init__(self):
        self.detector = PoseDetector(static_image_mode=False)
        self.counter = 0
        self.stage = None

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        results = self.detector.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                shoulder = [landmarks[12].x, landmarks[12].y]
                elbow = [landmarks[14].x, landmarks[14].y]
                wrist = [landmarks[16].x, landmarks[16].y]

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

            image = self.detector.draw_landmarks(image, results.pose_landmarks)

        return image

webrtc_streamer(key="bicep", video_processor_factory=BicepCounter)
