from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torchvision.transforms import functional as F, InterpolationMode

from torch import Tensor


class resnet50_feature():
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        model.to(self.device)
        model.eval()
        self.model = model
        self.preprocess = weights.transforms()

    def extract(self, img_path):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        img = read_image(img_path)    
        img = F.resize(img, 224, interpolation=InterpolationMode.BILINEAR, antialias=True)
        # img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=mean, std=std)
        batch = img.unsqueeze(0).to(self.device)

        #or VS
        # batch = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(batch)    

        npy_ = prediction.cpu().numpy()[0]
        return npy_


r50 = resnet50_feature()
# path = '/workspace/codes/playground/image_anomaly_detection/011setup/LVAD-Locally-Varying-Anomaly-Detection/img/1/10.png'
# feature = r50.extract(path)

import glob
target_folder = '/workspace/codes/playground/image_anomaly_detection/011setup/LVAD-Locally-Varying-Anomaly-Detection/img'
target_folder = '/home/ray/workspace/codes/playground/image_anomaly_detection/011setup/fish_LVAD-Locally-Varying-Anomaly-Detection/img'
file_list = glob.glob(target_folder + '/**/*.png', recursive=True)
file_list.sort()
import numpy as np
# data_store = np.empty((10, 2048))
data_store = {}
for i in range(1,11):
    name = f'{i}'
    data_store[name] = []

for file_ in file_list:
    class_name = file_.split('/')[-2]
    feature = r50.extract(file_)
    data_store[class_name].append(feature)
print('end')
