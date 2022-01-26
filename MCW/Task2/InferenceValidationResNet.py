'''
Author : Sarala Kumarage
ResNet101 Inference Validation using ImageNet dataset
Tensorflow 1.15 Compatible Code
'''
import glob
import logging
import pathlib

logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
assert float(tf.__version__[:3]) <= 1.15
print(tf.__version__)


from tensorflow.python.keras.applications.resnet import ResNet101
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.preprocessing import image
import numpy as np

def validate_resnet(data_dir, model, imagenet_class_ids):
    data_dir = pathlib.Path(data_dir)
    file_path = str(data_dir/'*/*')
    file_names_list = glob.glob(file_path)

    correct = 0
    count = 0
    acc = 0
    for file_path in file_names_list:
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        top1_output = decode_predictions(preds, top=1)[0]
        predcited_class=top1_output[0][0]
        parts = file_path.split('\\')
        one_hot = parts[-2] == imagenet_class_ids
        argmax = np.argmax(one_hot)
        # Integer encode the label
        actual=imagenet_class_ids[argmax]
        if (predcited_class == actual):
            correct = correct + 1
        count = count + 1
    # calculate the accuracy
    acc = (correct / count) * 100
    print('Correctly Predicted:', correct)
    print('Images Count:', count)
    print('Validation Accuracy:', acc)

def main():
    model = ResNet101(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000
    )

    '''
    #1.place the ImageNetLabels.txt file and set the path accordingly
    Download from the following link
    https://github.com/SaralaSewwandi/image_net_dataset/blob/main/ImageNetLabels.txt
    '''
    imagenet_class_ids = np.loadtxt("D:\\ImageNetLabels.txt", usecols=0, dtype=str)

    '''
    #2.place the ImageNet validation data set folder path path accordingly - Data set should be comprised with the folders with ImageNet class Ids

    download the imagenet data set with class name folders
    so this script validated the model based on the decoded class name
    https://drive.google.com/drive/u/1/folders/10pJ28cmO2KfdDdfX9uC0iNZnMPSp9ZM8
    '''
    data_dir = 'D:\\smaller_imagenet_validation'
    validate_resnet(data_dir, model,imagenet_class_ids)

if __name__ == "__main__":
    main()


