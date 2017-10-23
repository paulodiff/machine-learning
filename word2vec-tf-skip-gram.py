# http://www.thushv.com/natural_language_processing/word2vec-part-1-nlp-with-deep-learning-with-tensorflow-skip-gram/#skip-gram-model
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/5_word2vec.ipynb
# http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
#The goal of this assignment is to train a Word2Vec skip-gram model over Text8 data.


# %matplotlib inline

from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from sys import exit

# Download the data from the source website if necessary.

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

print('##### Downloading data')
filename = maybe_download('text8.zip', 31344016)

print('##### Read the data into a string.')

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)
print('Data size (number of words of text)  %d' % len(words))

print('##### Build the dictionary and replace rare words with UNK token')

vocabulary_size = 50000


myWords =['duck', 'duck', 'goose', 'duck', 'car', 'car']
myCount = [['UNK', -1]]
myCount.extend(collections.Counter(myWords).most_common(vocabulary_size - 1))

# print('SAMPLE: myWords:', myWords)
# print('SAMPLE: myCount', myCount)


def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

  # count contiene le prime (vocabulary_size parole) 500000 più frequenti del testo con il contatore a fianco
  # UNK Unknown (parole sconosciute) ...
  # [['UNK', -1], ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764), ('in', 372201), ('a', 325873), ......

  # print('count size %d' % len(count))
  # print('count sample data', count[:10])

  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)

  # print('dictionary size %d' % len(dictionary))
  # dictionary contiene tutte le parole e la loro posizione
  # {'morph': 37431, 'una': 19882, 'spicy': 17195, 'electronvolt': 48613,
  # print('dictionary data (electronvolt)', dictionary['electronvolt'])
  # print('dictionary data (spicy)', dictionary['spicy'])

  data = list()
  unk_count = 0

  # per ogni parola controlla che sia nel dizionario delle parole più frequenti (50000) ed aggiunge
  # l'indice del dizionario in "data" altrimenti incrementa il numero delle parole sconosciute
  # in count oltre ad avere il conteggio delle frequenze delle prime 50000 parole avremo anche il conteggio
  # delle parole sconosciute UNK

  # data contiene l'indice della parola del dizionario se appartiene al dizionario altrimenti 0 se è sconosciuta

  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)

  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)

# print('Most common words (+UNK) ... ', count[:10])

print('words size %d' % len(words))
print('words sample', words[:100])


print('data size %d' % len(data))
print('data sample', data[:100])

print('count size %d' % len(count))
print('count sample', count[:20])

print('dictionary size %d' % len(dictionary))
first2pairs = {k: dictionary[k] for k in sorted(dictionary.keys())[:20]}
print(first2pairs)

#for key in sorted(dictionary)[:10]:
#    print(key, ' - ', dictionary[key])

print('reverse_dictionary size %d' % len(reverse_dictionary))
# reverse_dictionary contiene la posizione e le parole
# {0: 'UNK', 1: 'the', 2: 'of', 3: 'and', 4: 'one', 5: 'in', 6: 'a', 7: 'to', 8: 'zero', 9: 'nine', 10: 'two', 11: 'is', 12: 'as', ...

# first2pairs = {k: reverse_dictionary[k] for k in reverse_dictionary.keys()[:2]}
first2pairs = {k: reverse_dictionary[k] for k in sorted(reverse_dictionary.keys())[:20]}
print(first2pairs)

#for key in sorted(reverse_dictionary)[:10]:
#    print(key, ' - ', reverse_dictionary[key])

# print('dictionary sample', dictionary)
# print('reverse_dictionary sample', reverse_dictionary)

del words  # Hint to reduce memory.




print('##### Function to generate a training batch for the skip-gram model')


data_index = 0

# Parametri batch_size dimensione del batch
#           num_skips
#           skip_window

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0 # sono divisibili
  assert num_skips <= 2 * skip_window # num_skips è almeno il doppio di skip_window

  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)

  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)

  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

my_batch_size = 20

print('data:', [reverse_dictionary[di] for di in data[:my_batch_size]])



for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=20, num_skips=num_skips, skip_window=skip_window)
    print('\nwith batch_size = %d num_skips = %d and skip_window = %d:' % (my_batch_size, num_skips, skip_window))
    print('    batch:', batch)
    print('    labels:', labels)
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(my_batch_size)])

# exit(0)

print('##### Train a skip-gram model. ')

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                   labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


num_steps = 100001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  average_loss = 0
  for step in range(num_steps):
    batch_data, batch_labels = generate_batch(
      batch_size, num_skips, skip_window)
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step, average_loss))
      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
  final_embeddings = normalized_embeddings.eval()

print('#### Plot data ')

num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])

def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()

words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)


