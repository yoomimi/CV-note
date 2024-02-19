import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.cm as cm

import os
from unet_model2 import UNet

model = UNet()  # Assuming UNet class definition is available
model_path = 'toyunet_model2.pth'
model.load_state_dict(torch.load(model_path))

test_images_path = os.path.join('test_images')

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Dice Score 계산 함수
def dice_score(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    intersection = (preds * labels).sum()
    dice = (2. * intersection) / (preds.sum() + labels.sum())
    return dice * 100

def load_and_transform_image(image_path, transform, device):

    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    return image




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

# 모델 평가
def evaluate_model(model, device, test_images_path, test_mask_dir, transform, output_dir):
    model.eval()  # 모델을 평가 모드로 설정
    os.makedirs(output_dir, exist_ok=True)  # 결과 저장 디렉토리 생성

    test_image_files = [f for f in os.listdir(test_images_path) if f.endswith('.png')]
    dice_scores = []  # Dice Score 저장을 위한 리스트

    with torch.no_grad():  # 그래디언트 계산 비활성화
        for image_file in test_image_files:
            image_path = os.path.join(test_images_path, image_file)
            pil_image = Image.open(image_path).convert('L')
            original_size = pil_image.size[::-1]  # (width, height) -> (height, width)
               
            image_tensor = transform(pil_image).unsqueeze(0).to(device)
            output = model(image_tensor)
            output = torch.sigmoid(output)
                
            # 예측된 마스크를 원본 이미지 크기로 업샘플링
            predicted = output > 0.5
            predicted_resized = F.interpolate(predicted.float(), size=original_size, mode='nearest').cpu().numpy().squeeze()
                
            # 실제 마스크 로딩 및 처리
            left_mask_path = os.path.join(test_mask_dir, 'L' + image_file)
            right_mask_path = os.path.join(test_mask_dir, 'R' + image_file)
            left_mask = np.array(Image.open(left_mask_path).convert('L')) / 255.0
            right_mask = np.array(Image.open(right_mask_path).convert('L')) / 255.0
            combined_mask = np.maximum(left_mask, right_mask)
                
            # Dice Score 계산
            dice = dice_score(predicted_resized.flatten(), combined_mask.flatten())
            dice_scores.append(dice)
                
            # 결과 저장
            seg_image = Image.fromarray((predicted_resized * 255).astype(np.uint8))
            seg_image.save(os.path.join(output_dir, image_file.replace('.png', '_pred.png')))


    # 평균 Dice Score 출력
    print(f'Dice Score: {dice_scores}')



# 모델, 디바이스 설정 및 평가 함수 호출
model.to(device)
transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 이미지를 (256, 256)으로 리사이징
        transforms.ToTensor()
    ])

output_dir = 'test_segmentation_results'
test_mask_dir = os.path.join('test_mask')  # 실제 마스크 이미지가 저장된 디렉토리 경로
evaluate_model(model, device, test_images_path, test_mask_dir, transform, output_dir)
