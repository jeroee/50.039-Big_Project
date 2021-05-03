# 50.039-Big_Project

Jeremy Ng 1003565

Zhou Yutong 1003704

Joey Richie L Tan 1003599 

Prashanth Nair 1003639

A deep learning project which focuses on detecting human emotion based on their facial features captured in videos by predicting the valence and arousal scores.

![](https://i.imgur.com/rMup4ix.gif)

## Instructions to run demo UI:

1. Clone directory

2. Download dataset from google drive. https://drive.google.com/drive/folders/1GhNLW-MWJbOHdfFbRV_HwPCm3P4kL6zi?usp=sharing unzip the folder and replace it with 'data' folder in repository

3. Copy path of any video from data>video>test_trim

4. pip install pyqt
 
5. Run python gui.py in terminal. A pyqt gui will pop up and prompt the user to insert full path of testing video. Click predict to view video frames. Hold 'k' to play the video and 


## Instructions to retrain model:

1. Clone directory

2. Download dataset from google drive. https://drive.google.com/drive/folders/1GhNLW-MWJbOHdfFbRV_HwPCm3P4kL6zi?usp=sharing unzip the folder and replace it with 'data' folder in repository

3. Change the paths in the Big_Project.ipynb and run all cells to train.

4. Model weights will be saved in 'saved weights' and validations loss will be saved in 'validation_results'.


## Credits

- Datasets taken from https://ibug.doc.ic.ac.uk/resources/aff-wild2/
- Implementation of Face Detection and Alignment using Multi-task Cascaded CNNs to https://github.com/TropComplique/mtcnn-pytorch





