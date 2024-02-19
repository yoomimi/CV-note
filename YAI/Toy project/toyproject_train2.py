import pandas as pd
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import colors
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.nn.functional import interpolate
from unet_model2 import UNet

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        self.images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.lower().endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.image_dir, image_name)
        # 왼쪽 폐 마스크와 오른쪽 폐 마스크 경로
        left_mask_path = os.path.join(self.mask_dir, 'L' + image_name)
        right_mask_path = os.path.join(self.mask_dir, 'R' + image_name)

        image = Image.open(image_path)
        left_mask = Image.open(left_mask_path).convert('L')
        right_mask = Image.open(right_mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            left_mask = self.transform(left_mask)
            right_mask = self.transform(right_mask)

        # 왼쪽과 오른쪽 마스크를 결합하여 하나의 마스크로 생성
        combined_mask = torch.max(left_mask, right_mask)

        # 이진 마스크로 변환
        mask = (combined_mask > 0.5).long().squeeze(0)  # Remove channel dim, since it's single channel
        return image, mask

  
# Google Drive 내의 경로 설정
image_dir = os.path.join('train_images')
left_mask_dir = os.path.join('train_mask')
right_mask_dir = left_mask_dir

# 이미지 리사이징 및 텐서 변환을 위한 transform 정의
transform = transforms.Compose([transforms.ToTensor()])

# Adjust the dataset and dataloader instantiation
train_dataset = SegmentationDataset(image_dir=image_dir, mask_dir=left_mask_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)



def resize_tensor(input_tensor, size=(256, 256), mode='nearest'):
    # 원래의 데이터 타입 저장
    original_dtype = input_tensor.dtype

    # interpolate 함수는 float 타입을 요구하기 때문에, 타입 변환
    input_tensor = input_tensor.float()

    # 배치 차원이 없으면 추가
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    elif input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    # 리사이즈 수행
    resized_tensor = interpolate(input_tensor, size=size, mode=mode)

    # 배치 차원이 추가된 경우 제거
    if resized_tensor.dim() == 4 and resized_tensor.size(0) == 1:
        resized_tensor = resized_tensor.squeeze(0)

    # 결과를 원래의 데이터 타입으로 되돌림
    resized_tensor = resized_tensor.to(original_dtype)

    return resized_tensor


def collate_fn(batch):
    images, target_masks = zip(*batch)

    # 이미지와 타겟 마스크를 텐서로 변환 및 리사이즈
    resized_images = [resize_tensor(image, size=(256, 256), mode='bilinear') for image in images]
    resized_target_masks = [resize_tensor(target_mask, size=(256, 256), mode='nearest') for target_mask in target_masks]

    # 텐서 리스트를 스택으로 변환
    images_batch = torch.stack(resized_images)
    target_masks_batch = torch.stack(resized_target_masks)

    return images_batch, target_masks_batch


# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)



# 모델 인스턴스 생성
model = UNet().to(device)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # 입력과 타겟을 평탄화(flatten)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

# 손실 함수 및 최적화 알고리즘 설정
criterion = DiceLoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001) # lr 0.0001로! (줄이는게 맞음)


import time

num_epochs = 100 # data 적으니까 이정도
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        # 모델 출력을 Sigmoid 활성화 함수를 통해 변환
        outputs = torch.sigmoid(outputs)

        # 출력 텐서의 크기를 타겟 마스크와 일치시키기
        # outputs의 크기를 masks와 동일하게 조정
        outputs_resized = F.interpolate(outputs, size=masks.size()[2:], mode='nearest')

        # Dice Loss 계산
        loss = criterion(outputs_resized, masks.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f} sec")



# Save the trained model
model_path = 'toyunet_model2.pth'
torch.save(model.state_dict(), model_path)
