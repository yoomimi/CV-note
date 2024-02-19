
# Lung segmentation with pytorch

### unet_model2.py

- 간단한 UNet model.

### toyproject_train2.py

- 40장 원본이미지 + 각 이미지당 왼쪽 폐 마스크 & 오른쪽 폐 마스크로 input dataset 구성하고 모델 학습 시키는 코드.
- Loss: Dice loss
- Optimizer: ADAM
- Learning rate: 0.0001
- Batch: 4
- Interpolating size: (256,256)

### toyproject_test2.py

- 5장으로 테스트. Test mask로 accuracy(dice accuracy)구하는 부분 구현.
- 결과는 segmentation_results 디렉토리에 저장되고 accuracy는 약 90%.




  
## 사용된 데이터 출처: Tuberculosis Chest X-ray Datasets

## Montgomery County CXR Set

The Montgomery County chest X-ray dataset is a collection of images aimed at supporting the development and evaluation of image segmentation and other image processing techniques for medical diagnosis, particularly for the detection and analysis of Tuberculosis.

### Dataset Overview

- **Source**: [National Library of Medicine](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html), part of the National Institutes of Health (NIH).
- **Content**: The dataset includes high-quality chest X-ray images along with segmentation masks that indicate regions of interest related to Tuberculosis findings.
- **Usage**: Primarily used for research in medical image processing, machine learning models for disease detection, and computer-aided diagnosis systems.

### Accessing the Dataset

You can download the dataset and its corresponding segmentation masks from the following link:

[Download Montgomery County CXR Set](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/index.html)

### Dataset Structure

- **Original Images**: High-resolution chest X-ray images.
- **Segmentation Masks**: Binary masks indicating the regions of Tuberculosis-related abnormalities.

### Usage License

Please refer to the provided link for detailed information on the dataset's license and usage restrictions. It is important to review these details to ensure compliance with the dataset's terms of use, especially for academic and research purposes.

