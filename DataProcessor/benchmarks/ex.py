import cv2
import os

p = "/home/ilya/Рабочий стол/TrendFlowML"

for file in os.listdir(p):

    if file.endswith(".mp4"):

        cap = cv2.VideoCapture(f"{p}/{file}")

        ok, frame = cap.read()

        print(file, frame.shape)