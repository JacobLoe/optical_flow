import numpy as np
import cv2


path = '../videos/unterordner/movies/Maltsev_Sem_Occupy_Wall_Street.mp4'

vid = cv2.VideoCapture(path)

ret, frame1 = vid.read()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

ret, frame2 = vid.read()
next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

ang = ang*180/np.pi/2

ang_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
mag_bins = [0, 20, 40, 60, 80, 100]

p = np.histogram(ang, bins=ang_bins)
q = np.histogram(mag, bins=mag_bins)

print(p)
print(q)
# print(ang)
# print(np.shape(mag), np.shape(ang))
# print(flow)
# print(np.shape(flow))
# print(np.shape(frame1))
