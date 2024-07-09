# emotion-detection-using-cnn


* First, clone the repository and enter the folder

```bash
git clone https://github.com/rebeccabas/emotion-detection-using-cnn.git
cd emotion-detection-using-cnn
```

* Download the FER-2013 dataset inside the `data` folder of `src` folder.

* If you want to train this model, use:  

```bash
cd src
python emotions.py --mode train
```

* To run the code after training, use:

```bash
cd src
python emotions.py --mode display
```


## Data Preparation 

* The [FER2013 dataset on Kaggle](https://www.kaggle.com/deadskull7/fer2013) is originally available as a single CSV file. The csv file is converted into PNG images for training and testing purposes.

* If you plan to try out new datasets, use data in CSV format. For convenience, `dataset_prepare.py` script is included which contains the code used for preprocessing the data. Feel free to use it as a reference.

## Algorithm

* Initially, the **Haar Cascade** technique is applied to detect faces in each frame of the webcam feed.

* The portion of the image containing the face is resized to **48x48** pixels and fed into the CNN.

* The network produces a list of **softmax scores** corresponding to the seven emotion classes.

* The emotion with the highest score is then displayed on the screen.

 
