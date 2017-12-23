# NIC Wrapper
The NIC Wrapper is a framework that augment the training set of [Google NIC](https://github.com/tensorflow/models/tree/master/research/im2txt#contact)
, the deep-learning based Neural Image Caption Generator. 

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

We intend to validate a hypothesis that extra training data from [Google Image](https://images.google.com/)
 may help [Google NIC](https://github.com/tensorflow/models/tree/master/research/im2txt#contact)
  learn the [MS COCO](http://cocodataset.org/#home) dataset better. NIC Wrapper is built for this purpose.
  During the training process of NIC, we keep inserting image caption pairs from Google as extra training samples.
  By doing this, we expect the mistakes made by NIC is corrected. 
  
This 
![Show and Tell Architecture](phase2.png)
Suppose NIC sees a picture 

 Suppose an image caption generator sees a picture
 of "dog" while describing it as "cat". The predicted description of "cat" is fed to a search engine,
 that suggests images of "cat". The image caption generator sees the suggested images, together with the query "cat", 
 that it realize a "cat" should look like a cat but not a dog. 

 
 NIC Wrapper keeps adding images caption pairs from Google 

<!--- Comments are Fun 
This project aims to augment the training set of the Google NIC. Initially the NIC model 
is trained on MS COCOCO caption 
 by keeps sourcing new samples from an search engine.
 --->
 
 
 We propose a framework that intends to improve the deep-learning based Neural Image Caption,or NIC, 
 by providing extra training samples from search engines. 
  While the NIC is being trained  on the COCO benchmark  dataset,
 we  keep  inserting  image  caption  pairs  sourced from the Google Image as extra training samples. 
.
 

## Contact
Zitong Lian (LEAAN | lianzitong@yahoo.com)





How to use
To train:
When train on the MS COCO, run train.py
When train on both MS COCO and Google images, run train_wrapper.py

To evaluate:
run evaluate.py
While the model is being trained.

To insert Google images:
run 