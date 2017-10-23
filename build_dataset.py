# https://www.codementor.io/jimmfleming/how-i-built-a-reverse-image-search-with-machine-learning-and-tensorflow-part-1-8dje8gjm9
# https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

# Build DataSet ...
# Da una cartelle di immagini prepara il dataset per la rete
#
# 1. Rotazione destra sinistra (3°)
# 2. Traslazione 10, 20 px
# 3. Zoom
import PIL
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
import skimage

# REMOVE WARNING compile!
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from skimage import data
from skimage import io
from skimage import transform
from skimage.transform import rescale
import tensorflow as tf
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)
tf.logging.set_verbosity(tf.logging.INFO)


print('1-Start --------------------------------------------------------- 1 ')

SOURCE_FILE_FOLDER = 'C:/Users/M05831/Downloads/C_I/output_png'
SOURCE_FILE_FOLDER_COLLECTION = 'C:/Users/M05831/Downloads/C_I/output_png/*.png'
OUTPUT_FILE_FOLDER = 'C:/Users/M05831/Downloads/C_I/output_test'

# TF https://www.tensorflow.org/programmers_guide/reading_data
# Reading from files
# A typical pipeline for reading records from files has the following stages:
# The list of filenames
# Optional filename shuffling
# Optional epoch limit
# Filename queue
# A Reader for the file format
# A decoder for a record read by the reader
# Optional preprocessing
# Example queue

# Make a queue of file names including all the JPEG images files in the relative
# image directory.


def generate_input_fn(file_pattern, batch_size, num_epochs=None, shuffle=False):
    "Return _input_fn for use with Experiment."

    def _input_fn():
        height, width, channels = [256, 256, 3]

        filenames_tensor = tf.train.match_filenames_once(file_pattern)
        filename_queue = tf.train.string_input_producer(
            filenames_tensor,
            num_epochs=num_epochs,
            shuffle=shuffle)

        reader = tf.WholeFileReader()
        filename, contents = reader.read(filename_queue)

        image = tf.image.decode_jpeg(contents, channels=channels)
        image = tf.image.resize_image_with_crop_or_pad(image, height, width)
        image_batch, filname_batch = tf.train.batch(
            [image, filename],
            batch_size,
            num_threads=4,
            capacity=50000)

        # Converts image from uint8 to float32 and rescale from 0..255 => 0..1
        # Rescale from 0..1 => -1..1 so that the "center" of the image range is roughly 0.
        image_batch = tf.to_float(image_batch) / 255
        image_batch = (image_batch * 2) - 1

        features = {
            "image": image_batch,
            "filename": filname_batch
        }

        labels = {
            "image": image_batch
        }

        return features, labels
    return _input_fn




all_files = [ (SOURCE_FILE_FOLDER + '/' + f) for f in os.listdir(SOURCE_FILE_FOLDER) if os.path.isfile(os.path.join(SOURCE_FILE_FOLDER, f))]
print(all_files)

print('2-Start --------------------------------------------------------- 1 ')
fq1 = tf.train.match_filenames_once(all_files)
# fq1 = tf.train.match_filenames_once(['./images/*.jpg','c:/*.*'])
#fq1 = ["c:/Users/M05831/PycharmProjects/nn/images/a.png", "c:/Users/M05831/PycharmProjects/nn/images/b.png"]

#tf.train.match_filenames_once('')

print('3-Start --------------------------------------------------------- 1 ')
filename_queue = tf.train.string_input_producer(all_files)

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
print('4-Start --------------------------------------------------------- 1 ')
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
print('5-Start --------------------------------------------------------- 1 ')
_, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
# image = tf.image.decode_jpeg(image_file)   JPEG
print('6-Start --------------------------------------------------------- 1 ')
image = tf.image.decode_png(image_file)

init = tf.global_variables_initializer()

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    #tf.initialize_all_variables().run()

    sess.run(init)


    F,L = sess.run([generate_input_fn])

    print('R1 - Start --------------------------------------------------------- 1 ')
    # print(sess.run(fq1))

    print('R2 - Start --------------------------------------------------------- 2 ')
    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print('R3 - Start --------------------------------------------------------- 2 ')
    # Get an image tensor and print its value.
    image_tensor = sess.run([image])
    print(image_tensor)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)



