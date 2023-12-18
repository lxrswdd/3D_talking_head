import face_alignment
import cv2
import os
from os.path import join
import numpy as np
from tqdm import tqdm
import json
from glob import glob
import argparse
#requires face_alignment=1.4.0

class Gen2DLandmarks(object):
    def __init__(self) -> None:
        super().__init__()
        self.fa_func = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

        
    def main_process(self, img_dir,out_dir):
        # out_dir = "/data/share/CelebA-HQ/landmarks"
        
        #img_path_list = [x for x in glob("%s/*.png" % img_dir) if "mask" not in x]
        img_path_list = [x for x in glob("%s/*.jpg" % img_dir) if "mask" not in x]
        
        if len(img_path_list) == 0:
            print("Dir: %s does include any .png images." % img_dir)
            exit(0)
        
        img_path_list.sort()

        for img_path in tqdm(img_path_list, desc="Generate facial landmarks"):
            
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            res = self.fa_func.get_landmarks(img_rgb)
            
            if res is None:
                print("Warning: can't predict the landmark info of %s" % img_path)
                
            # base_name = img_path[img_path.rfind("/") + 1:-4]

            save_name = os.path.basename(img_path)[:-4] + "_lm2d.txt"
            save_path = os.path.join(out_dir, save_name)
            try:
                preds = res[0]
                np.save(os.path.join(out_dir,os.path.basename(img_path)[:-4]),preds)
            except:
                print("Warning: can't predict the landmark info of %s" % img_path)
                with open(os.path.join(out_dir,'fail_list.txt'), "a") as f:
                    f.write("Warning: can't predict the landmark info of %s" % img_path)
                continue    
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='The code for generating facial landmarks.')
    # parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    tt = Gen2DLandmarks()
    tt.main_process(args.img_dir)
    