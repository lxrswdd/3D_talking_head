"""
This is a fake model
"""

from math import log
import numpy as np
import cv2

class FakeModel:
    def __init__(self, model_path):
        self.fake_model = np.load(model_path, allow_pickle=True).item()
        self.pitch_range = 0.16
        self.yaw_range = 0.46
        self.step = 0.01
        self.decimal = int(log(1/self.step, 10))
        self.last_img = np.zeros((512,512,3), dtype=np.float32)
        print('model initialized')
    def render(self, pitch, yaw):
        # pitch -0.16 ~ 0.16
        # yaw   -0.46 ~ 0.46
        # ------------------------
        # returns an image of 512x512x3 dtype: np.float32
        
        pitch = round(pitch, self.decimal)
        yaw = round(yaw, self.decimal)
        if pitch < - self.pitch_range: pitch = - self.pitch_range
        if pitch > self.pitch_range: pitch = self.pitch_range
        if yaw < - self.yaw_range: yaw = - self.yaw_range
        if yaw > self.yaw_range: yaw = self.yaw_range
        if pitch == 0: pitch = 0.0
        if yaw == 0: yaw = 0.0
                
        if f'{pitch},{yaw}' in self.fake_model:
            img = np.array(self.fake_model[f'{pitch},{yaw}'], dtype=np.float32)
            img = img / 255.
            img = cv2.resize(img, (512, 512))
            self.last_img = img
        else:
            print(f'Current pitch: {pitch}, yaw: {yaw} are out of range.')
            # print('NO')
            img = self.last_img
            
        return img
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create and load the model
    model = FakeModel(model_path=r"D:\OneDrive - UBC\Projects\3D_HEAD\UI\models\fake_model.npy")
    
    # render an image
    img =  model.render(pitch=0, yaw=0)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', img)
    cv2.waitKey()

    # plt.imshow(img)
    # plt.show()
