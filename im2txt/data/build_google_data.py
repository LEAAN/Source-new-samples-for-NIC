# coding=utf-8
'''
This script keeps crawling images from Google so that the model keeps seeing extra training samples.
While the train_wrapper.py runs, this script predicts captions for a proportion of mscoco training images
 using the latest model checkpoint. These predcited captions are fed to Google, whose suggested images are
 added to the folder of training data, together with their textual queries as the ground truth.
In the next epoch of training, the model sees both mscoco data and data from Google as training data.
The Google data is renewed every epoch during training.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse
import glob
import os
from PIL import Image
import logging
import time
from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
from im2txt import utils
from collections import namedtuple
from icrawler.builtin import GoogleImageCrawler
from im2txt.data import build_mscoco_data


ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "captions"])

parser = argparse.ArgumentParser()
parser.add_argument('data_folder',
                    help='Path to the folder where the training data, the model checkpoints, and the word count file locate.',
                    default='/mnt/raid/data/ni/dnn/zlian/')
args = parser.parse_args()

data_folder=args.data_folder
# use the latest ckpt to source images from Google
# otherwise pass the path to a specific ckpt in the data_folder
# ckpt_folder= "/mnt/raid/data/ni/dnn/zlian/ckpt-3-milli/model.ckpt-3000000"
ckpt_foler = data_folder+"ckpt-1-milli"

# This word_count file should not be modified or replace. Otherwise the index of words in the whole vocabulary change.
# If missing, plz download from the github user psycharo via: https://github.com/tensorflow/models/issues/466
vocab_file=data_folder+'word_counts.txt'

# Directory for saving images returned by Google, corresponds to the output_dir in build_mscoco_data.py,
# by default = "/mnt/raid/data/ni/dnn/zlian/Google_image"
google_file_folder= data_folder+"Google_image/"

# A flag that marks the training process.
flag_file = google_file_folder+"flag.txt"

# Where coco tfrecords are saved
coco_folder = data_folder+"mscoco/"

# Where Google imgs sourced for the current epoch are saved
current_folder = None

def predict_images(filenames, vocab, n_sentences =2):
    """
    Use the latest model checkpoint to predict (infer) captions for part of the training images.
       ilenames: list of image filenames to infer
       n_sentence: number of sentences generated for each iamge, max=3

       return: list of captions predicted by the most recent ckpt. Each caption shall be a string
       eg: predict_seqs = [["I", "wish","to","get","rid","of","acne"],[******]]
       The real captions to be used in Imagemetadata is different.
       captions=[[u"<S>","I", "wish","to","get","rid","of","acne",".","</S>"]]
    """
    print ('Using ckpt {0} to infer'.format(ckpt_foler))
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   ckpt_foler)
    g.finalize()

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)
        generator = caption_generator.CaptionGenerator(model, vocab)
        predict_seqs = []
        for filename in filenames:
            with tf.gfile.GFile(filename, "r") as f:
                image = f.read()
            captions = generator.beam_search(sess, image)
            if (len(captions)< n_sentences):
                n_sentences = len(captions)
            for i in range(n_sentences):
                # Ignore begin and end words. sentence is a list of words
                sentence = [vocab.id_to_word(w) for w in captions[i].sentence[1:-1]]
                sentence = " ".join(sentence)
                predict_seqs.append(sentence)

        global_step = model.global_stepp.eval()
        print ('global step is {0} :D'.format(global_step))

    global current_folder
    current_folder =google_file_folder + str(global_step) +'/'
    utils.createfolder(current_folder)
    savingname= current_folder+ 'pred_seqs.pkl'
    utils.save(predict_seqs, savingname, ('Predicted seqs are saved to %s :D') % savingname)
    print ('total number of pred_seqs: %d' %len(predict_seqs))
    return savingname


def png2jpg(png_files):
    '''
    Change png files to jpg files.
    '''
    for png in png_files:
        im = Image.open(png)
        rgb_im = im.convert('RGB')
        rgb_im.save(png[:-4]+'.jpg')
        os.remove(png)
        print (png + ' is saaaaaaaved :D')


def jPG2jpg(jPG_files):
    '''
    Change image file names from jPG to jpg.
    '''
    for jPG in jPG_files:
        os.rename(jPG, jPG[:-4]+'.jpg')
        print(jPG + ' is saaaaaaaved :D')


def crawl_images(queries, n_google):
    """
    Crawl images from Google
    Queries: A list of strings that are queries used to source Google images.
    n_google: return top n results from google
    """
    image_metadata = []
    counter = 0
    for query in queries:
        try:
            current_dir = current_folder + str(counter)
            google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=2,
                                                log_level=logging.ERROR,
                                                storage={'root_dir':current_dir})
            google_crawler.crawl(keyword=query, offset=0, max_num=n_google,
                                 date_min=None, date_max=None,
                                 min_size=(300, 300), max_size=None)
            query = query.split()
            pngs = glob.glob(current_dir + "/*.png")+ glob.glob(current_dir + "/*.PNG")
            JPGs = glob.glob(current_dir + "/*.JPG") + glob.glob(current_dir + "/*.JPEG")
            png2jpg(png_files=pngs)
            jPG2jpg(jPG_files=JPGs)
            images = glob.glob(current_dir + "/*.*jpg")
            for image in images:
                captions = [["<S>"] + query + ["</S>"]]
                image_metadata.append(ImageMetadata(counter, image, captions))
            # Save metadata every 100 record :D
            if not counter%100:
                utils.save(image_metadata, current_folder+ 'metadata_template_%d.pkl'%counter, '\n')

        except:
            print ('Abandon folder %s' %(current_folder + str(counter)))
        counter+=1

    print ('Metadata len %d' %len(image_metadata))
    print ('counter should be 100, actually it is {0}'.format(counter))
    savingname = current_folder+ 'metadata.pkl'
    utils.save(image_metadata, savingname, '\n')
    return savingname

def clearImg():
    '''
    Clean tfrecords used for this epoch, before adding new tfrecords for the next epoch.
    '''
    try:
        os.system('rm %strain-*****-of-00008' % (coco_folder))
        print('Existing images from Google are removed :D')
    except:
        pass
    # google_file_folder should be the path where TF-records of Google images locate.
    os.system('cp %strain* %s' % (google_file_folder, current_folder))
    os.system('cp %strain* %s' % (google_file_folder, coco_folder))
    os.system('rm %strain*' % google_file_folder)


def main():
    # Number of images whose predicted captions will be used to retrieve Google images.
    n_infer=630
    # Number of top results from Google to be used.
    n_google = 10
    # Number of sentences to be inferred for each selected image. Equals to the beam size.
    n_sentences = 3
    print ('Randomly select {0} images, predict {1} captions for each, return top {2} results from Google'.format(n_infer, n_google, n_sentences))

    # Mscoco_train_dataset + mscoco_val_dataset[0:train_cutoff]
    input_file_folder = coco_folder+"raw-data/train2014/"
    train_filenames = glob.glob(input_file_folder + "/*.jpg")
    # Randomly selected training images to infer.
    rand = np.random.randint(len(train_filenames), size=n_infer)
    images_rand = [train_filenames[i] for i in rand]

    while True:
        flag=utils.readflag(path=flag_file)
        if flag!=2:
            # The flag has set to 1 or 0 by train_wrapper.py
            vocab = vocabulary.Vocabulary(vocab_file)

            # Predict the captions for some images.
            print ('Predicting image captions. May take a while.')
            seqpath = predict_images(filenames=images_rand, vocab = vocab, n_sentences=n_sentences)

            # Prediction is finished.
            predict_seqs = utils.load(seqpath, 'Predicted seqs are loaded from %s' % seqpath)
            print ('len of predicted_seqs %s' %len(predict_seqs))

            # Crawl images using the predicted captions.
            print ('crawling images. May take a long time.')
            metapath = crawl_images(predict_seqs, n_google = n_google)
            metadata = utils.load(metapath, 'Metadata is loaded from %s' % metapath)

            # Save raw Google images as tfrecords.
            # May vary the num_shards so that each tfrecord saves around 2300 image caption pairs.
            build_mscoco_data._process_dataset("train", metadata, vocab, num_shards=8)

            # Remove tfrecords used in the last epoch.
            clearImg()

            # Set flag=2 and the train_wrapper.py knows the crawler has finished retrieving images for the next epoch.
            utils.writeflag(path=flag_file, flag=2, info='New images are ready for a new training')

            # Give some time for the train_wrapper.py to write flag=0, and start a new training.
            time.sleep(300)
        else: time.sleep(300)

if __name__=='__main__':
    main()

