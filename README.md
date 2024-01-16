# ML_IDCard_Segmentation (Tensorflow / Keras)
Machine Learning Project to identify an ID Card on an image.  

### Objectives
The goal of this project is to recognize a ID Card on a photo, cut it out using semantic segmentation and to 
transform the perspective so that you get a front view of the ID Card.
Optionally an OCR text recognition can be done in a later step.
However, this is not yet planned in the scope of this project.

## Additional Information
Dataset: [MIDV-500](https://arxiv.org/abs/1807.05786)   


## Installation
1. Create and activate a new environment.
```
conda create -n idcard python=3.8
conda activate idcard
```
2. Install Dependencies.
```
pip install -r requirements.txt
```

## Download and Prepare Dataset
Download the image files (image and ground_truth).  
Splits the data into training, test and validation data.
```
python prepare_dataset.py
```

### Training of the neural network
```
python train.py
```

### Show Jupyter Notebook for Test
```
jupyter notebook "IDCard Prediction Test.ipynb"
```

### Test the trained model
```
python test.py test/sample1.png --output_mask=test/output_mask.png --output_prediction=test/output_pred.png --model=model.h5
```


## Background Information

### Model
A [U-NET](https://arxiv.org/abs/1505.04597) was used as the model.
The network is based on the fully convolutional networkand its architecture was modified and extended to work with
fewer training images and to yield more precise segmentations. 
Segmentation of a 512*512 image takes less than a second on a modern GPU.



## Results for validation set (only trained on german id cards)
Accuracy:  
94.87%




