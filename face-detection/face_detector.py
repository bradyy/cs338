from mtcnn import MTCNN 
import cv2

class Face_detector:
    def  __init__(self) :
        self.detector=MTCNN()
    def detect(self,img):
        result=self.detector.detect_faces(img)
        if len(result)==0:
            return None
        bbox =result[0]["box"] # get bbox
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[0] + bbox[2]
        y_max = bbox[1] + bbox[3]

        return (x_min, y_min, x_max, y_max)