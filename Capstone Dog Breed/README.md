# Dog Breed Classifier-DSND Capstone



## Project Overview

In this project, you will learn how to build a pipeline that  can be used within a web or mobile app to process real-world,  user-supplied images.  Given an image of a dog, your algorithm will  identify an estimate of the canine’s breed.  If supplied an image of a  human, the code will identify the resembling dog breed.

[![Sample Output](https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/sample_dog_output.png)](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/project-dog-classification/images/sample_dog_output.png)

Along with exploring state-of-the-art CNN models for classification  and localization, you will make important design decisions about the  user experience for your app.  Our goal is that by completing this lab,  you understand the challenges involved in piecing together a series of  models designed to perform various tasks in a data processing pipeline.   Each model has its strengths and weaknesses, and engineering a  real-world application often involves solving many problems without a  perfect answer.  Your imperfect solution will nonetheless create a fun  user experience!



## Import Datasets

* Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
* Download the [human_dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)



## Analysis
As showcased in the jupyter notebook, I have below analysis results of the input dataset:

* There are 133 total dog categories.
* There are 8351 total dog images.
* There are 6680 training dog images.
* There are 835 validation dog images.
* There are 836 test dog images.
* There are 13233 total human images.

-----
## Project Results:
In summary, the performance of the pre-trained model I built far exceeded the hand made CNN model. The accuracy of the model reached 80% while my CNN was about 13%.  This improved performance can be attributed to the vast dataset on which the pre-trained model was built.  In particular, the pre-trained model was also exposed to many dog images, making it particularly ready to classify dog breeds.

Link to Medium story here: https://medium.com/@mukhtar.rad/dog-vs-human-detectability-and-similarity-40c5a68d9575





