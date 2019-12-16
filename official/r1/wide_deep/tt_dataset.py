from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# pylint: disable=wrong-import-order
from absl import app as absl_app
from absl import flags
from six.moves import urllib
from six.moves import zip
import pandas as pd
import tensorflow as tf
# pylint: enable=wrong-import-order

from official.utils.flags import core as flags_core




DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult'
TRAINING_FILE = 'df_train_10k.csv'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_FILE = 'df_eval_3k.csv'
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

# data_dir = 'C:\\tmp\\a6_nowait_data'
# os.path.join(data_dir, TRAINING_FILE)
# os.path.join(data_dir, EVAL_FILE)


_CSV_COLUMNS = [
    'speed', 'ais_rem_time', 'inv_speed', 'haversine_distance', 'coastal_rem_time',
    'dest_lat', 'dest_lon', 'dest_country_code',
    'y_actual_log'
]

_CSV_COLUMN_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0],
                        [0.0], [0.0], [''],
                        [0.0]]

_HASH_BUCKET_SIZE = 1000

_NUM_EXAMPLES = {
    'train': 29814,
    'validation': 9934,
}


def _download_and_clean_file(filename, url):
  """Downloads data from url, and makes changes to match the CSV format."""
  temp_file, _ = urllib.request.urlretrieve(url)
  with tf.compat.v1.gfile.Open(temp_file, 'r') as temp_eval_file:
    with tf.compat.v1.gfile.Open(filename, 'w') as eval_file:
      for line in temp_eval_file:
        line = line.strip()
        line = line.replace(', ', ',')
        if not line or ',' not in line:
          continue
        if line[-1] == '.':
          line = line[:-1]
        line += '\n'
        eval_file.write(line)
  tf.io.gfile.remove(temp_file)


def download(data_dir):
  """Download census data if it is not already present."""
  tf.io.gfile.makedirs(data_dir)

  training_file_path = os.path.join(data_dir, TRAINING_FILE)
  if not tf.io.gfile.exists(training_file_path):
    _download_and_clean_file(training_file_path, TRAINING_URL)

  eval_file_path = os.path.join(data_dir, EVAL_FILE)
  if not tf.io.gfile.exists(eval_file_path):
    _download_and_clean_file(eval_file_path, EVAL_URL)


def build_model_columns():
  """
  :param data = df_preprocess_noNA
  Builds a set of wide and deep feature columns.
  """
  # Continuous variable columns
  speed = tf.feature_column.numeric_column('speed')
  inv_speed = tf.feature_column.numeric_column('inv_speed')
  ais_rem_time = tf.feature_column.numeric_column('ais_rem_time')
  haversine_distance = tf.feature_column.numeric_column('haversine_distance')
  coastal_rem_time = tf.feature_column.numeric_column('coastal_rem_time')
  dest_lat = tf.feature_column.numeric_column('dest_lat')
  dest_lon = tf.feature_column.numeric_column('dest_lon')

  # df_train_10k['dest_lat'] = df_train_10k['dest_lat'].astype(str)
  # df_train_10k['dest_lon'] = df_train_10k['dest_lon'].astype(str)
  # dest_lat_levels_ls = df_train_10k['dest_lat'].unique().tolist()
  # dest_lon_levels_ls = df_train_10k['dest_lon'].unique().tolist()

  # categorical_column_with_vocabulary_list: OHE, use for low cardinality catg fts
  # use embeddings for high cardinality (i.e. > thousands unique levels) catg fts
  # when cardinality is very high, it becomes infeasible to train a neural network using OHE
  # embedding: low dim dense vector, need to tune embedding size
  # sch_scac = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'sch_scac', sch_scac_levels_ls)
  dest_country_code = tf.feature_column.categorical_column_with_vocabulary_list(
      'dest_country_code', ['IL',
                         'ZA',
                         'OT',
                         'LY',
                         'PT',
                         'BE',
                         'ES',
                         'NL',
                         'CN',
                         'DE',
                         'ID',
                         'JP',
                         'AU',
                         'GB',
                         'US',
                         'CA',
                         'TR',
                         'KR',
                         'PA',
                         'IN',
                         'AR',
                         'IT',
                         'FR',
                         'OM',
                         'EG',
                         'MG',
                         'RU',
                         'MA',
                         'SG',
                         'EC',
                         'AE',
                         'TW',
                         'RE',
                         'HK',
                         'CO',
                         'CR',
                         'NZ'])
  # dest_lat = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'dest_lat', dest_lat_levels_ls)
  # dest_lon = tf.feature_column.categorical_column_with_vocabulary_list(
  #     'dest_lon', dest_lon_levels_ls)

  # use _hash_bucket when sparse features are in string or integer format
  # A hash function takes as input a key, which is associated with a record and used to identify it to data storage and retrieval application.
  # The keys may be fixed length, like an integer, or variable length, like a name.
  # The output is a hash code used to index a hash table holding the data or records, or pointers to them.
  # Set hash_bucket_size to avoid collisions: https://stackoverflow.com/questions/45685301/principle-of-setting-hash-bucket-size-parameter
  # To show an example of hashing:
  # occupation = tf.feature_column.categorical_column_with_hash_bucket(
  #     'occupation', hash_bucket_size=_HASH_BUCKET_SIZE)

  # Transformations.
  # age_buckets = tf.feature_column.bucketized_column(
  #     age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  # Wide columns and deep columns.
  base_columns = [dest_country_code,
                  ]

  # crossed_columns = [
  #     tf.feature_column.crossed_column(
  #         ['education', 'occupation'], hash_bucket_size=_HASH_BUCKET_SIZE),
  #     tf.feature_column.crossed_column(
  #         [age_buckets, 'education', 'occupation'],
  #         hash_bucket_size=_HASH_BUCKET_SIZE),
  # ]
  # Create cross fts for any catg column, except categorical_column_with_hash_bucket (since crossed_column hashes the input)
  # Keep uncrossed fts in model, indep ft help model to distinguish btwn samples where a collision occurred in crossed ft
  # wide_columns = base_columns + crossed_columns

  wide_columns = base_columns

  deep_columns = [
      speed,
      inv_speed,
      ais_rem_time,
      haversine_distance,
      coastal_rem_time,
      dest_lat,
      dest_lon,
      # tf.feature_column.indicator_column(sch_scac),
      tf.feature_column.indicator_column(dest_country_code),
      # tf.feature_column.indicator_column(dest_lat),
      # tf.feature_column.indicator_column(dest_lon),
      # To show an example of embedding
      # tf.feature_column.embedding_column(occupation, dimension=8),
  ]

  return wide_columns, deep_columns


# data_dir = 'C:\\tmp\\a6_nowait_data'
# train_file = os.path.join(data_dir, TRAINING_FILE)
# eval_file = os.path.join(data_dir, EVAL_FILE)


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.io.gfile.exists(data_file), (
      '%s not found. Please make sure you have run a6_nowait_dataset.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  def parse_csv(value):
    tf.compat.v1.logging.info('Parsing {}'.format(data_file))
    columns = tf.io.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(list(zip(_CSV_COLUMNS, columns)))
    labels = features.pop('y_actual_log')
    return features, labels

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file).skip(1)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  # set of examples used in one iteration (that is, one gradient update) in training
  # mini-batch size usually 10-1000; fixed during training and inference.
  return dataset


def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", default="C:\\tmp\\a6_nowait_data",
      help=flags_core.help_wrap(
          "Directory to download and extract data."))


def main(_):
  download(flags.FLAGS.data_dir)


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  define_data_download_flags()
  absl_app.run(main)
