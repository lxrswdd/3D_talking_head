import os
import sys
import torch
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from IPython import display as ipydisplay
from tqdm import tqdm
import cv2
from time import time
from math import sqrt
from torchvision.transforms import transforms
# os.chdir("/data/share/Code/SegmentTool")
sys.path.append('/data/share/Code/SegmentTool/face_parsing')
from face_parsing.model import BiSeNet

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print('CUDA is available. Device: ', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('CUDA is NOT available. Use CPU instead.')
     

# Parsing Net
segnet = BiSeNet(n_classes=19)
segnet.cuda()
segnet.load_state_dict(torch.load('./face_parsing/79999_iter.pth'))
segnet.eval()

def run_face_parsing(img):
    # input standardized to 0 ~ 1
    n_classes = 19
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        img = to_tensor(img)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = segnet(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    ###################
    ## Release CUDA
    import gc
    gc.collect() # colloct memory
    torch.cuda.empty_cache() # empty cuda
    return parsing


def class_remapping(parsing_mask):
    """
    Old Classes:
        atts = {1: 'skin', 2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye', 
                6: 'eye_g', 7: 'l_ear', 8: 'r_ear', 9: 'ear_r', 10: 'nose', 
                11: 'mouth', 12: 'u_lip', 13: 'l_lip', 14: 'neck', 15: 'neck_l', 
                16: 'cloth', 17: 'hair', 18: 'hat'}
    New Classes (11 classes):
            0: background
            1: face skin
            2: eye brows
            3: eyes
            4: nose
            5: upper lip
            6: lower lip
            7: ears
            8: hair
            9: hat
            10: eyeglasses
            11: mouth
    """
    #new_mask = np.copy(silhouette).astype(np.int)
    new_mask = np.zeros(parsing_mask.shape).astype(np.int)
    def process(parsing_mask, old_class, new_class):
        one_mask = np.where(parsing_mask == old_class, 1, 0)
        return one_mask, new_class*one_mask
    mapping = { 1: 1,
                2: 2, 
                3: 2, 
                4: 3, 
                5: 3, 
                10: 4, 
                12: 5, 
                13: 6,
                7: 7,
                8: 7,
                17: 8,
                18: 9,
                6: 10,
                11: 11} # format  old_class: new_class
    for old_class in mapping.keys():
        one_mask, class_mask = process(parsing_mask, old_class=old_class, new_class=mapping[old_class])
        new_mask = new_mask * (1 - one_mask) + class_mask    
    return new_mask



def read_img(img_path, resize=512):
    # 3-channel image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize, resize))
    img = np.array(img, dtype=np.float32) / 255. # 0 ~ 1
    return img


def read_mask(mask_path, resize=512):
    # binary mask 8-bit
    mask = cv2.imread(mask_path, 0) / 255
    mask = cv2.resize(mask, (resize, resize))
    return mask
    
def apply_mask(img, mask):
    # set backgound color to gray
    img_copy = np.copy(img) * 2 - 1
    for c in range(img_copy.shape[2]):
        img_copy[:,:,c] = img_copy[:,:,c] * mask
    return np.clip(img_copy / 2 + 0.5, 0, 1)

def gen_face_mask(img_folder_dir,out_dir):
    img_list = os.listdir(img_folder_dir)
    img_list = [os.path.join(img_folder_dir,i) for i in img_list if i.endswith('.jpg')]
    for img_dir in tqdm(img_list):
        img = read_img(img_dir)
        basename = os.path.basename(img_dir)[:-4]
        parsing = run_face_parsing(img)
        mask = np.zeros(img.shape[:2], dtype=np.float32)
        mask[ np.where( (parsing >= 1) & (parsing <= 13) & (parsing != 6) ) ] = 1
        mask2 = cv2.resize(mask, (256, 256))
        np.save(os.path.join(out_dir,basename+'.npy'),mask2)
