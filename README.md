# Source-new-samples-for-NIC
This project aims to augment the training set of an image caption generator by keeps sourcing new samples from an search engine.


## Contact
Zitong Lian (LEAAN | lianzitong@yahoo.com)

## Contents
* [Overview](#overview)
* [Requirement](#getting-started)
    * [A Note on Hardware and Training Time](#a-note-on-hardware-and-training-time)
    * [Install Required Packages](#install-required-packages)
    * [Prepare the Training Data](#prepare-the-training-data)
    * [Download the Inception v3 Checkpoint](#download-the-inception-v3-checkpoint)
* [Installation](#installation)
* [Train](#training-a-model)
    * [Initial Training](#initial-training)
    * [Fine Tune the Inception v3 Model](#fine-tune-the-inception-v3-model)
    * [Image Insertion](#image-insertion)
* [Evaluate](#evaluate)


### Overview
This 
![Show and Tell Architecture](phase2.png)


How to use
To train:
When train on the MS COCO, run train.py
When train on both MS COCO and Google images, run train_wrapper.py

To evaluate:
run evaluate.py
While the model is being trained.

To insert Google images:
run 