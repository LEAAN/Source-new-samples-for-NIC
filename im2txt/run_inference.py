# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import json

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
from datetime import datetime

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "/mnt/raid/data/ni/dnn/zlian/ckpt-1-milli/model.ckpt-1000000",
                       "model checkpoint file or directory containing a model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "/mnt/raid/data/ni/dnn/zlian/ckpt-1-milli/word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "/mnt/raid/data/ni/dnn/zlian/mscoco/raw-data/val2014/COCO_val2014_000000224477.jpg",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)


def predict4all(addedflag='latest_ckpt'):
  '''
  Predict captions for all validation images using the latest model ckpt.
  add "predict4all(addedflag=addedflage)" in main when called.
  :param addedflag: a flag use to name the json file that saves all prediction results
  '''
  print ('The predict4all function infers all val images using the ckpt defined in FLAG.checkpoint_path, \n'
         'which should be changed everytime'
         )
  # The path to the dir that saves mscoco valiation images.
  imgDir = "/mnt/raid/data/ni/dnn/zlian/mscoco/raw-data/val2014/"

  # The path to the file that saves annotations fo mscoco images.
  captions_file = '/mnt/raid/data/ni/dnn/zlian/mscoco/raw-data/annotations/captions_val2014.json'
  with tf.gfile.FastGFile(captions_file, "r") as f:
    caption_data = json.load(f)

  # Create map from id to filename (391895, u'COCO_val2014_000000391895.jpg')
  filename_to_id={}
  for x in caption_data['images']:
    filename_to_id[x["file_name"]]= x["id"]

  filenames=filename_to_id.keys()
  print ('***********')
  print ('Added flag is {0}'.format(addedflag))
  print ('Filenames look like {0}'.format(filenames[0]))
  print ('This is the length of filename_to_id {0}'.format(len(filename_to_id)))

  # Create the saving list
  results=[]

  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  with tf.Session(graph=g) as sess:
    restore_fn(sess)
    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    generator.beam_size=1
    counter=0

    for filename in filenames:
      try:
        # Some imgs cannot be opened.
        with tf.gfile.GFile(imgDir+filename, "r") as f:
          image = f.read()
        caption = generator.beam_search(sess, image)[0]

        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print (sentence)
        counter += 1
      except Exception as e:
        print (e)

      if not counter % 100:
        print (str(datetime.now()))
        print ('{0} images are predicted'.format(counter))
        print ('An example image caption pair looks like this: {0} --- {1}'.format(filename, sentence))

      # [{"image_id": 404464, "caption": "black and white photo of a man standing in front of a building"},...]
      img_id=filename_to_id[filename]
      results.append({'image_id':img_id, 'caption':sentence})

  # Save results as json file
  savingName=(FLAGS.checkpoint_path[FLAGS.checkpoint_path.find('model.ckpt-')+6:]+'_'+addedflag)
  print ('Saving Name is {0}'.format(savingName))
  savingPath='/mnt/raid/data/ni/dnn/zlian/coco_eval/results/predicted_captions_'+savingName
  print ('Saving prediction results to ... {0}'.format(savingPath))
  with open(savingPath, 'w') as outfile:
    json.dump(results, outfile)


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(" "):
    filenames.extend(tf.gfile.Glob(file_pattern))

  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    for filename in filenames:
      with tf.gfile.GFile(filename, "r") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      print("Captions for image %s:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))



if __name__ == "__main__":
  tf.app.run()
