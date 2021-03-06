import os
from absl import app as absl_app
from absl import flags
import tensorflow as tf
import pickle

from official.utils.flags import core as flags_core
from official.utils.logs import logger
from official.r1.wide_deep import tt_dataset # if using models/official/r1/wide_deep branch
# from notebooks.model.cseta_a6_nowait import tt_dataset
from official.r1.wide_deep import wide_deep_run_loop
print('Remember to empty model directory (D:\\tmp\\a6_nowait_model) before running main.py')
print('Also, set project interpreter to tensorflow_37, or 3.7 okay; this uses 3.7')


def define_a6_nowait_flags():
  wide_deep_run_loop.define_wide_deep_flags()
  flags.adopt_module_key_flags(wide_deep_run_loop)
  flags_core.set_defaults(data_dir='D:\\tmp\\a6_nowait_data',
                          model_dir='D:\\tmp\\a6_nowait_model_sampled_noevalstep',
                          train_epochs=500,
                          epochs_between_evals=2,
                          inter_op_parallelism_threads=0,
                          intra_op_parallelism_threads=0,
                          batch_size=40)


def build_estimator(model_dir, model_type, model_column_fn, inter_op, intra_op):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = model_column_fn()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.compat.v1.estimator.RunConfig().replace(
      session_config=tf.compat.v1.ConfigProto(device_count={'GPU': 0},
                                              inter_op_parallelism_threads=inter_op,
                                              intra_op_parallelism_threads=intra_op))
  if model_type == 'wide':
    return tf.compat.v1.estimator.LinearRegressor(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.compat.v1.estimator.DNNRegressor(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.compat.v1.estimator.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def run_tt(flags_obj):
  """Construct all necessary functions and call run_loop.

  Args:
    flags_obj: Object containing user specified flags.
  """
  # if flags_obj.download_if_missing:
  #   tt_dataset.download(flags_obj.data_dir)


  train_file = os.path.join(flags_obj.data_dir, tt_dataset.TRAINING_FILE)
  test_file = os.path.join(flags_obj.data_dir, tt_dataset.EVAL_FILE)
  def train_input_fn():
    return tt_dataset.input_fn(
        train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size)

  def eval_input_fn():
    return tt_dataset.input_fn(test_file, 1, False, flags_obj.batch_size)

  tensors_to_log = {
      'average_loss': '{loss_prefix}head/truediv',
      'loss': '{loss_prefix}head/weighted_loss/Sum'
  }

  results = wide_deep_run_loop.run_loop(
      name="Transit Time", train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      model_column_fn=tt_dataset.build_model_columns,
      build_estimator_fn=build_estimator,
      flags_obj=flags_obj,
      tensors_to_log=tensors_to_log,
      early_stop=True)

  wide_deep_run_loop.export_model(model = results['model'], model_type='wide_deep',
                                  export_dir = 'D:\\tmp\\a6_nowait_tfmodel_sampled_noevalstep',
                                  model_column_fn=tt_dataset.build_model_columns)
  print('Creating list of predictions...')
  eval_pred_ls = list(results['eval_pred'])
  train_pred_ls = list(results['train_pred'])
  print('Finished creating list of predictions.')
  print('Dumping eval predictions to pickle file...')
  with open('D:\\tmp\\a6_nowait_predictions_sampled_noevalstep\\eval_pred.txt', 'wb') as fp:
      pickle.dump(eval_pred_ls, fp)
  print('Dumping train predictions to pickle file...')
  with open('D:\\tmp\\a6_nowait_predictions_sampled_noevalstep\\train_pred.txt', 'wb') as fp:
      pickle.dump(train_pred_ls, fp)
  print('Finished dumping predictions to pickle files.')



def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_tt(flags.FLAGS)


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  define_a6_nowait_flags()
  absl_app.run(main)

# View tensorboard
# conda activate tensorflow_37
# (tensorflow_37) C:\Users\DUJE2\gvvmc_new\gvvmc>tensorboard --logdir='C:\\tmp\\a6_nowait_model'

# save model
# with tf.Session(graph=tf.Graph()) as sess:
#     imported = tf.compat.v1.saved_model.load(sess, ["serve"], "C:\\tmp\\a6_nowait_tfmodel\\1576564864")
