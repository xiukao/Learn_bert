# coding=utf-8
"""Fine-tune BERT for SWAG four-choice sentence completion (CPU/GPU)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import random

import experiment_utils
import modeling
import optimization
import tokenization
import numpy as np
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None, "Directory with train.csv, val.csv, test.csv.")
flags.DEFINE_string("bert_config_file", None, "BERT model configuration JSON.")
flags.DEFINE_string("vocab_file", None, "BERT WordPiece vocabulary.")
flags.DEFINE_string("output_dir", None, "Directory for this run.")
flags.DEFINE_string("init_checkpoint", None, "Pre-trained BERT checkpoint prefix.")
flags.DEFINE_bool("do_lower_case", True, "Match the selected checkpoint.")
flags.DEFINE_integer("max_seq_length", 128, "Maximum WordPiece sequence length.")
flags.DEFINE_bool("do_train", False, "Train on SWAG train.csv.")
flags.DEFINE_bool("do_eval", False, "Evaluate on SWAG val.csv.")
flags.DEFINE_bool("do_predict", False, "Predict SWAG test.csv.")
flags.DEFINE_integer("random_seed", 42, "Python, NumPy and TensorFlow seed.")
flags.DEFINE_integer("train_batch_size", 16, "Training examples per batch.")
flags.DEFINE_integer("eval_batch_size", 8, "Evaluation examples per batch.")
flags.DEFINE_integer("predict_batch_size", 8, "Prediction examples per batch.")
flags.DEFINE_float("learning_rate", 2e-5, "Initial Adam learning rate.")
flags.DEFINE_float("num_train_epochs", 3.0, "Training epochs.")
flags.DEFINE_float("warmup_proportion", 0.1, "Learning-rate warmup fraction.")
flags.DEFINE_integer("save_checkpoints_steps", 1000, "Checkpoint interval.")
flags.DEFINE_bool("use_tpu", False, "SWAG runner currently supports CPU/GPU.")


def read_examples(path, has_labels):
  """Read official SWAG CSV columns; one CSV row is one four-choice example."""
  examples = []
  with open(path, "r", encoding="utf-8", newline="") as stream:
    reader = csv.DictReader(stream)
    required = ["sent1", "sent2", "ending0", "ending1", "ending2", "ending3"]
    if has_labels:
      required.append("label")
    missing = [name for name in required if name not in reader.fieldnames]
    if missing:
      raise ValueError("Missing SWAG columns in %s: %s" %
                       (path, ", ".join(missing)))
    for row_number, row in enumerate(reader):
      context = tokenization.convert_to_unicode(row["sent1"])
      stem = tokenization.convert_to_unicode(row["sent2"])
      endings = [
          tokenization.convert_to_unicode(row["ending%d" % i])
          for i in range(4)]
      if not context or not stem or any(not ending for ending in endings):
        raise ValueError("Empty SWAG text at row %d in %s" %
                         (row_number + 2, path))
      label = int(row["label"]) if has_labels else 0
      if label < 0 or label > 3:
        raise ValueError("Invalid SWAG label at row %d" % (row_number + 2))
      examples.append((context, stem, endings, label))
  if not examples:
    raise ValueError("No SWAG examples in %s" % path)
  return examples


def convert_choice(context, continuation, tokenizer, max_seq_length):
  tokens_a = tokenizer.tokenize(context)
  tokens_b = tokenizer.tokenize(continuation)
  while len(tokens_a) + len(tokens_b) > max_seq_length - 3:
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()
  tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
  segment_ids = ([0] * (len(tokens_a) + 2) +
                 [1] * (len(tokens_b) + 1))
  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask = [1] * len(input_ids)
  padding = max_seq_length - len(input_ids)
  return (input_ids + [0] * padding,
          input_mask + [0] * padding,
          segment_ids + [0] * padding)


def write_tfrecord(examples, tokenizer, max_seq_length, path):
  writer = tf.python_io.TFRecordWriter(path)
  for context, stem, endings, label in examples:
    choices = [convert_choice(
        context, stem + " " + ending, tokenizer, max_seq_length)
               for ending in endings]
    values = {
        "input_ids": [choice[0] for choice in choices],
        "input_mask": [choice[1] for choice in choices],
        "segment_ids": [choice[2] for choice in choices],
        "label_ids": [label],
    }
    features = {}
    for name, value in values.items():
      flattened = ([item for choice in value for item in choice]
                   if name != "label_ids" else value)
      features[name] = tf.train.Feature(
          int64_list=tf.train.Int64List(value=flattened))
    record = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(record.SerializeToString())
  writer.close()


def input_fn_builder(path, max_seq_length, batch_size, is_training):
  feature_spec = {
      "input_ids": tf.FixedLenFeature([4, max_seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([4, max_seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([4, max_seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
  }

  def decode(record):
    example = tf.parse_single_example(record, feature_spec)
    return {name: tf.cast(value, tf.int32)
            for name, value in example.items()}

  def input_fn():
    dataset = tf.data.TFRecordDataset(path)
    if is_training:
      dataset = dataset.shuffle(100).repeat()
    dataset = dataset.map(decode)
    return dataset.batch(batch_size, drop_remainder=False)

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, max_seq_length):
  def model_fn(features, labels, mode, params):
    del labels, params
    batch_size = tf.shape(features["input_ids"])[0]
    flat_ids = tf.reshape(features["input_ids"], [-1, max_seq_length])
    flat_mask = tf.reshape(features["input_mask"], [-1, max_seq_length])
    flat_segments = tf.reshape(features["segment_ids"], [-1, max_seq_length])
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    model = modeling.BertModel(
        config=bert_config, is_training=is_training, input_ids=flat_ids,
        input_mask=flat_mask, token_type_ids=flat_segments,
        use_one_hot_embeddings=False)
    pooled = model.get_pooled_output()
    hidden_size = pooled.shape[-1].value
    if is_training:
      pooled = tf.nn.dropout(pooled, keep_prob=0.9)
    weights = tf.get_variable(
        "swag_output_weights", [hidden_size, 1],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    bias = tf.get_variable(
        "swag_output_bias", [1], initializer=tf.zeros_initializer())
    logits = tf.reshape(tf.matmul(pooled, weights) + bias, [batch_size, 4])
    probabilities = tf.nn.softmax(logits, axis=-1)
    label_ids = features["label_ids"]
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_ids, logits=logits)
    loss = tf.reduce_mean(losses)

    if init_checkpoint:
      assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
          tf.trainable_variables(), init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, False)
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
      predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
      accuracy = tf.metrics.accuracy(label_ids, predictions)
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops={"accuracy": accuracy})
    return tf.estimator.EstimatorSpec(
        mode=mode, predictions={"probabilities": probabilities})

  return model_fn


def main(_):
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)
  tf.set_random_seed(FLAGS.random_seed)
  if FLAGS.use_tpu:
    raise ValueError("run_swag.py currently supports CPU/GPU only")
  if not (FLAGS.do_train or FLAGS.do_eval or FLAGS.do_predict):
    raise ValueError("Select --do_train, --do_eval, or --do_predict")
  if FLAGS.max_seq_length < 4:
    raise ValueError("max_seq_length must be at least 4")
  tokenization.validate_case_matches_checkpoint(
      FLAGS.do_lower_case, FLAGS.init_checkpoint)
  tf.gfile.MakeDirs(FLAGS.output_dir)
  experiment_utils.save_effective_flags(FLAGS, FLAGS.output_dir)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError("max_seq_length exceeds BERT position embeddings")
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = read_examples(
        os.path.join(FLAGS.data_dir, "train.csv"), True)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    if num_train_steps < 1:
      raise ValueError("Training configuration produces zero steps")
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    experiment_utils.save_computed(FLAGS.output_dir, {
        "train_examples": len(train_examples),
        "num_train_steps": num_train_steps,
        "num_warmup_steps": num_warmup_steps})
  estimator = tf.estimator.Estimator(
      model_fn=model_fn_builder(
          bert_config, FLAGS.init_checkpoint, FLAGS.learning_rate,
          num_train_steps, num_warmup_steps, FLAGS.max_seq_length),
      model_dir=FLAGS.output_dir,
      config=tf.estimator.RunConfig(
          save_checkpoints_steps=FLAGS.save_checkpoints_steps))

  if FLAGS.do_train:
    train_record = os.path.join(FLAGS.output_dir, "train.tf_record")
    write_tfrecord(train_examples, tokenizer, FLAGS.max_seq_length,
                   train_record)
    estimator.train(
        input_fn=input_fn_builder(train_record, FLAGS.max_seq_length,
                                  FLAGS.train_batch_size, True),
        max_steps=num_train_steps)

  if FLAGS.do_eval:
    dev_examples = read_examples(
        os.path.join(FLAGS.data_dir, "val.csv"), True)
    dev_record = os.path.join(FLAGS.output_dir, "eval.tf_record")
    write_tfrecord(dev_examples, tokenizer, FLAGS.max_seq_length, dev_record)
    results = estimator.evaluate(input_fn=input_fn_builder(
        dev_record, FLAGS.max_seq_length, FLAGS.eval_batch_size, False))
    with open(os.path.join(FLAGS.output_dir, "eval_results.txt"), "w") as stream:
      for name in sorted(results):
        stream.write("%s = %s\n" % (name, results[name]))

  if FLAGS.do_predict:
    test_examples = read_examples(
        os.path.join(FLAGS.data_dir, "test.csv"), False)
    test_record = os.path.join(FLAGS.output_dir, "predict.tf_record")
    write_tfrecord(test_examples, tokenizer, FLAGS.max_seq_length,
                   test_record)
    predictions = estimator.predict(input_fn=input_fn_builder(
        test_record, FLAGS.max_seq_length, FLAGS.predict_batch_size, False))
    with open(os.path.join(FLAGS.output_dir, "test_results.tsv"), "w") as stream:
      for prediction in predictions:
        stream.write("\t".join(
            str(value) for value in prediction["probabilities"]) + "\n")


if __name__ == "__main__":
  for required_flag in ("data_dir", "bert_config_file", "vocab_file",
                        "output_dir"):
    flags.mark_flag_as_required(required_flag)
  tf.app.run()
