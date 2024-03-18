from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("best.pt")

im1 = cv2.imread("i.jpg")
results = model.predict(source=im1) 

im2 = cv2.imread("o.jpg")
results = model.predict(source=im2)  

