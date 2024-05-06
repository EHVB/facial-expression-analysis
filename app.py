import cv2
import streamlit as st
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av


# st.set_page_config(layout="wide")

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
classifier =load_model('model.keras')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']


# class Faceemotion(VideoTransformerBase):
#     def recv(self, frame):
#         img = frame.to_ndarray(format="bgr24")

#         #image gray
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(img_gray, scaleFactor=1.1, minSize=(200, 200), minNeighbors=5,flags=cv2.CASCADE_SCALE_IMAGE)
        
#         for (x, y, w, h) in faces:
#             cv2.rectangle(img=img, pt1=(x,y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)
#             roi_gray = img_gray[y:y + h, x:x + w]
#             roi_gray = cv2.resize(roi_gray, (56, 56), interpolation=cv2.INTER_AREA)
            
#             if np.sum([roi_gray]) != 0:
#                 roi = roi_gray.astype('float') / 255.0
#                 roi = img_to_array(roi)
#                 roi = np.expand_dims(roi, axis=0)
                
#                 prediction = classifier.predict(roi)[0]
#                 label=emotion_labels[prediction.argmax()]
                
#             label_position = (x, y)
#             cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         return img
    
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(img_gray, scaleFactor=1.1, minSize=(200, 200), minNeighbors=5,flags=cv2.CASCADE_SCALE_IMAGE)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x,y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (56, 56), interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                
            label_position = (x, y)
            cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        # video_processor_factory=Faceemotion,
        video_frame_callback= video_frame_callback)

    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.confidence_threshold = 0.5
        
if __name__ == "__main__":
    main()