# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from im2txt import configuration
from im2txt import show_and_tell_model
from im2txt import utils
import time


FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("input_file_pattern", "/mnt/raid/data/ni/dnn/zlian/mscoco/train-?????-of-?????",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_boolean("train_inception", True,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("log_every_n_steps", 100,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_string("train_dir",
                       "/mnt/raid/data/ni/dnn/zlian/ckpt-1-milli/",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("inception_checkpoint_file",
                       "/mnt/raid/data/ni/dnn/zlian/inception_v3.ckpt",
                        "Path to a pretrained inception_v3 model.")

# Path to the file that saves a flag marking the training stage of NIC while inserting images from Google.
flag_file = "/mnt/raid/data/ni/dnn/zlian/Google_image/flag.txt"

def train(number_of_steps):
  model_config = configuration.ModelConfig()
  model_config.input_file_pattern = FLAGS.input_file_pattern
  training_config = configuration.TrainingConfig()
  model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file

  # Create training directory.
  train_dir = FLAGS.train_dir
  # if not tf.gfile.IsDirectory(train_dir):
  #   tf.logging.info("Creating training directory: %s", train_dir)
  #   tf.gfile.MakeDirs(train_dir)

  # Build the TensorFlow graph.
  g = tf.Graph()
  with g.as_default():
    # Build the model.
    model=show_and_tell_model.ShowAndTellModel(
        model_config, mode="train", train_inception=FLAGS.train_inception)
    model.build()

    # Set up the learning rate.
    learning_rate_decay_fn = None
    if FLAGS.train_inception:
      print ("The inception weights are fine-tuned together with weights in the LSTM units and word embeddings.")
      learning_rate = tf.constant(training_config.train_inception_learning_rate)
    else:
      print ("The inception weights are frozen. Only weights in the LSTMs and word embeddings are randomly"
             "initialized and trained.")
      learning_rate = tf.constant(training_config.initial_learning_rate)
      if training_config.learning_rate_decay_factor > 0:
        num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                 model_config.batch_size)
        decay_steps = int(num_batches_per_epoch *
                          training_config.num_epochs_per_decay)

        def _learning_rate_decay_fn(learning_rate, global_step):
          return tf.train.exponential_decay(
              learning_rate,
              global_step,
              decay_steps=decay_steps,
              decay_rate=training_config.learning_rate_decay_factor,
              staircase=True)

        learning_rate_decay_fn = _learning_rate_decay_fn

    # Set up the training ops.
    train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=training_config.optimizer,
        clip_gradients=training_config.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)
    # saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

  # Run training.
  tf.contrib.slim.learning.train(
      train_op,
      train_dir,
      log_every_n_steps=FLAGS.log_every_n_steps,
      graph=g,
      global_step=model.global_step,
      number_of_steps=number_of_steps,
      init_fn=model.init_fn,
      saver=saver)

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    # Number of steps per epoch when batch_size=32
    steps_per_epoch = 18323
    # The step from where we start our training from existing model ckpts,
    # either from other github users or from our own.
    #  Set to zero when train from scratch.
    min_step=1000000

    # flag=1 marks the stage when the current epoch of training is taking place.
    utils.writeflag(path=flag_file, flag=1, info='Train for a few steps')

    # Train a few steps with only coco data.
    train(number_of_steps=min_step+10)

    # flag=0 marks the stage when the current epoch of training is stopped, either finished or paused.
    utils.writeflag(path=flag_file, flag=0, info='Stop train for a few steps')
    # Here the current epoch of training is paused until the crawler finishes crawling Google images for the next epoch.
    # The crawler is defined in im2txt/data/build_google_data.py
    time.sleep(600)

    # Train with Google images for 10 more epochs.
    min_step +=steps_per_epoch*1
    steps=[min_step]
    n_loops = 10
    i = 0
    while(i<n_loops):
        steps.append(steps[i]+steps_per_epoch)
        i+=1
    print ('Train the model until these steps one by one: {0}'.format(steps))

    for step in steps:
        # The current epoch of training starts
        utils.writeflag(path=flag_file, flag=1, info='start training')
        print ("Train until step %d" %step)
        train(number_of_steps=step)

        # The current epoch of training finishes.
        utils.writeflag(path=flag_file, flag=0, info='finish training, wait for the crawler')
        while True:
            flag = utils.readflag(path=flag_file)
            # flag=2 marks the stage when this epoch of training is finished,
            # and the crawler has also finished crawling Google images for the next epoch.
            if flag==2:
                # write the flag back to 0, and start the next epoch of training and image insertion.
                utils.writeflag(path=flag_file, flag=0, info='Crawling is done. Move to the next step :D')
                break
            else:   time.sleep(600)

    print ("Eventually the training process has finished. \n"
           "Please terminate the crawler by killing build_google_data.py. \n"
           "Please also terminate the evaluation process by killing evaluate.py")


if __name__ == "__main__":
  tf.app.run()
