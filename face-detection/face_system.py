
from multiprocessing import process
import os
import pickle
from PIL import Image
import numpy as np
import cv2
from scipy import spatial
import time
import glob
from tqdm import tqdm
import json

import feature_extractor
import face_detector

class Facesystem:
    def __init__(self,dataset_path,extractor_method):
        if extractor_method=="VGG":
            self.extractor=feature_extractor.VGG16_FE()
        elif extractor_method=="Xception":
            self.extractor=feature_extractor.Xception_FE()
        elif extractor_method=="VGGFACE":
            self.extractor=feature_extractor.VGGFACE_FE()
        self.detector=face_detector.Face_detector()
        self.dataset_path = dataset_path
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        self.image_folder = os.path.join(self.dataset_path,'images')

        self.feature_folder_path = os.path.join(self.dataset_path,'feature')
        if not os.path.exists(self.feature_folder_path):
            os.mkdir(self.feature_folder_path)
    def index(self):
        for img_path in tqdm(os.listdir(self.image_folder)):
            #print("hello word")
            #print(img_path)
            name = img_path.split('/')[-1][:-3]
            vector_file = os.path.join(self.feature_folder_path,name+'.pkl')
            img_path_full = os.path.join(self.image_folder+'\\'+img_path)
            #print(img_path_full)
            img = cv2.imread(img_path_full)
            if self.detector.detect(img) is None:
                return 0
            x_min, y_min, x_max, y_max = self.detector.detect(img)
            PIL_image = Image.open(img_path_full).crop((x_min, y_min, x_max, y_max))
            try:
                feature_vector = self.extractor.extract(PIL_image)
            except:
                continue
            pickle.dump(feature_vector,open(vector_file,'wb'))
    def recognition(self,img):
        res_dict={}
        
        top_similarity=0.0
        for vector_file in os.listdir(self.feature_folder_path):
            vector_file_path=os.path.join(self.feature_folder_path,vector_file)
            vector=pickle.load(open(vector_file_path,'rb'))
            similarity=1-spatial.distance.cosine(vector,self.extractor.extract(img))
            name=vector_file.split('.')[0]
            temp = {'similary': similarity}
            res_dict[name] = temp

            if similarity> top_similarity:
                top_similarity = similarity
                face_name = vector_file.split('.')[0]

        if top_similarity < 0.5:
            return 'Unknown'

        return face_name, top_similarity, res_dict
    def recognition_1(self,img,vectors):
        res_dict={}
        
        top_similarity=0.0
        i=0
        for vector_file in os.listdir(self.feature_folder_path):
            #vector_file_path=os.path.join(self.feature_folder_path,vector_file)
            vector=vectors[i]
            similarity=1-spatial.distance.cosine(vector,self.extractor.extract(img))
            print(similarity)
            name=vector_file.split('.')[0]
            i+=1
            if similarity> top_similarity:
                face_name = vector_file.split('.')[0]
                top_similarity=similarity
        if top_similarity < 0.5:
            return 'Unknown'

        return face_name    
    def recognition_img(self,img_path):
        if isinstance(img_path, str):
            PIL_img = Image.open(img_path)
            img = cv2.imread(img_path)
        x_min,y_min,x_max,y_max=self.detector.detect(img)
        process_img=PIL_img.crop((x_min, y_min, x_max, y_max))
        bbox_img=cv2.rectangle(img,(x_min,y_min),(x_max,y_max),(0,255,0),2)
        if (self.recognition(process_img))[0]=='Unknown':
            face_name = "Unknown"
            cv2.putText(bbox_img, face_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            face_name = (self.recognition(process_img))[0]
            cv2.putText(bbox_img, face_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Result", bbox_img)
        cv2.waitKey(0)
        cv2.imwrite('result.jpg', bbox_img)
    def recognition_webcam(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        vectors=[]
        for vector_file in os.listdir(self.feature_folder_path):
            vector_file_path=os.path.join(self.feature_folder_path,vector_file)
            vector=pickle.load(open(vector_file_path,'rb'))
            vectors.append(vector)

        while True:
            ret, frame = cap.read()

            if ret is None:
                continue

            # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            if self.detector.detect(frame) is None:
                continue
            

            x_min, y_min, x_max, y_max = self.detector.detect(frame)
            processed_frame = Image.fromarray(np.uint8(frame[y_min:y_max, x_min:x_max])).convert('RGB')
            res_frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            if self.recognition_1(processed_frame,vectors) == "Unknown":
                face_name = "Unknown"
                cv2.putText(res_frame, face_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                face_name, similarity, _ = self.recognition(processed_frame)
                cv2.putText(res_frame, f'{face_name}: {round(similarity, 2)}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Result", res_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
system=Facesystem('datasets','VGGFACE')
#system.index()
#system.recognition_img("duong.jpg")
system.recognition_webcam()