from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from bleu import BLEU
import data_utils
import seq2seq_model
import jieba
import re
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 10,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("from_vocab_size", 49178, "English vocabulary size.")
tf.app.flags.DEFINE_integer("to_vocab_size", 49178, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/data/repos_ysf/mul_dialogue_mul_layer/data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/data/repos_ysf/mul_dialogue_mul_layer/modelns", "Training directory.")
tf.app.flags.DEFINE_string("from_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data", None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data", None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data", None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 20),(10,30),(15,40),(20,50)]

def nor(float_num):
    return float(int(float_num*100)/100)

def max_array(a):
    return np.argmax(a)

def cal_distance(x,y):
    return ((x-y)**2).sum()

def get_bucket_id(quetions,answer):
    max_id = -1
    for source_ids in quetions:
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(answer) < target_size:
                return  bucket_id
    if max_id == -1:
        return 3
    else:
        return max_id
def read_mul_data(source_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:

      source,target= source_file.readline(),source_file.readline()
      counter = 0
      question = []
      question.append(source)
      while source and target and (not max_size or counter < max_size):
        #print(question)
        if len(question)>3:
            question.pop(0)

        counter += 1
        if counter % 20000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_idss = []
        source_idss1 =[]
        source_idss1.append([])
        source_idss1.append([])
        source_idss1.append([int(x) for x in question[-1].split()])
        for i in range(3-len(question)):
            source_idss.append([])
        for sx in question:
            source_ids = [int(x) for x in sx.split()]
            source_idss.append(source_ids)

        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        bucket_id = get_bucket_id(question,target_ids)
        bucket_id1 = get_bucket_id([question[-1]],target_ids)
        '''
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        '''
        data_set[bucket_id].append([source_idss, target_ids])
        data_set[bucket_id1].append([source_idss1, target_ids])
        question.append(target)
        target = source_file.readline()
        source = question[-1]
        s_ids = [int(x) for x in target.split()]
        if 8 in s_ids and 7 in s_ids:
            source, target = source_file.readline(), source_file.readline()
            question =[]
            question.append(source)


      for x in range(4):
          print(len(data_set[i]))
  return data_set
def read_data(source_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:

      source = source_file.readline()
      target =source
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 20000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break

        source = source_file.readline()
        target = source

  return data_set

def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.from_vocab_size,
      FLAGS.from_vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt :
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model

def train():
  """
  if FLAGS.from_train_data and FLAGS.to_train_data:
    from_train_data = FLAGS.from_train_data
    to_train_data = FLAGS.to_train_data
    from_dev_data = from_train_data
    to_dev_data = to_train_data
    if FLAGS.from_dev_data and FLAGS.to_dev_data:
      from_dev_data = FLAGS.from_dev_data
      to_dev_data = FLAGS.to_dev_data
    from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_data(
        FLAGS.data_dir,
        from_train_data,
        to_train_data,
        from_dev_data,
        to_dev_data,
        FLAGS.from_vocab_size,
        FLAGS.to_vocab_size)
  else:
      # Prepare WMT data.
      print("Preparing WMT data in %s" % FLAGS.data_dir)
      from_train, to_train, from_dev, to_dev, _, _ = data_utils.prepare_wmt_data(
          FLAGS.data_dir, FLAGS.from_vocab_size, FLAGS.to_vocab_size)
  """
  #data_path_pre = '/data/repos_ysf/mul_dialogue_mul_layer/data/'
  from_dev = os.path.join(FLAGS.data_dir,'train_id_test.in')#data_path_pre + 'train_id_test.in'
  from_train = os.path.join(FLAGS.data_dir,'train_id.in')

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    print(from_train)
    dev_set = read_mul_data(from_dev)
    train_set = read_mul_data(from_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    bleu_count = 0
    bleu = 0
    index =0
    while index <50000:
      index = index + 1
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights, bleu_answer = model.get_batch(
          train_set, bucket_id)
      _, step_loss, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      #print(len(output_logits),len(output_logits[0]),len(output_logits[0][0]))
      #print(output_logits)max_array
      #print(step_loss,type(step_loss))
      '''
      outputs =[]
      for i in range(len(output_logits)):
          out_put=[]
          for j in range(len(output_logits[0])):
              out_put.append(max_array(output_logits[i][j]))
          outputs.append(out_put)
      '''
      #cal bleu begin
      '''
      #outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      output_candicate=[]
      for i in range(len(outputs[0])):
          hhh=[]
          for j in range(len(outputs)):
              hhh.append(outputs[j][i])
          output_candicate.append(hhh)
      for i in range(len(output_candicate)):
          if data_utils.EOS_ID in output_candicate[i]:
              output_candicate[i] = output_candicate[i][:output_candicate[i].index(data_utils.EOS_ID)]

      if len(output_candicate) == len(bleu_answer):
          for i in range(len(output_candicate)):
              candidate = output_candicate[i]
              reference = bleu_answer[i]
              candidate = [str(k) for k in candidate]
              reference = [str(k) for k in reference]
              cstr = ' '.join(candidate)
              rstr = ' '.join(reference)
              candidate =[]
              reference =[]
              candidate.append(cstr)
              reference.append(rstr)
              #print(cstr,'QA',rstr)
              references = []
              references.append(reference)
              bleu = bleu+nor(BLEU(candidate, references))
              bleu_count = bleu_count + 1
      else:
          print('there some problem please look look')
      #cal bleu end
      '''
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        #print('bleu', bleu / bleu_count)
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights, _ = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _= model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          #print(eval_loss,type(eval_loss))
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()
    print('50000 is ok')

def top_k(x,k):
    y = np.argsort(x)
    return y[-k:]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def rm_pun(sencen):
    sen = sencen.strip()
    sen = re.sub('[+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+', ' ', sen)
    sen = jieba.cut(sen, cut_all=False)
    return ' '.join(sen)

def decode():

  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.
    # Load vocabularies.
    en_vocab_path = os.path.join(FLAGS.data_dir,'train_dict.in')

    en_vocab, rev_fr_vocab = data_utils.initialize_vocabulary(en_vocab_path)


    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    questions =[]
    while sentence:
      # Get token-ids for the input sentence.
      #seg_list = jieba.cut(sentence, cut_all=False)
      #sentence =" ".join(seg_list)
      sentence = rm_pun(sentence)
      print(sentence)
      token_ids = data_utils.sentence_to_token_ids(sentence, en_vocab)

      questions.append(token_ids)
      if len(questions) > 3:
          questions.pop(0)
      source_idss = []
      for i in range(3 - len(questions)):
          source_idss.append([])
      for sx in questions:
          source_idss.append(sx)
      bucket_id = get_bucket_id(questions, [])

      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights ,_= model.get_batch(
          {bucket_id: [(source_idss, [])]}, bucket_id)
      # Get output logits for the sentence.
      state,_, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      '''
      state =state[0]
      #print( len(output_logits), len(output_logits[0]), len(output_logits[0][0]))
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      sentence_list = data_model.get_answer(sentence)
      sen_list =[]
      for sen in sentence_list:
          token_ids = data_utilsn.sentence_to_token_ids(rm_pun(sen[0]), en_vocab)
          # Which bucket does it belong to?
          bucket_id = len(_buckets) - 1
          for i, bucket in enumerate(_buckets):
              if bucket[0] >= len(token_ids):
                  bucket_id = i
                  break
          
          else:
              logging.warning("Sentence truncated: %s", sentence)
          
          # Get a 1-element batch to feed the sentence to the model.
          encoder_inputs, decoder_inputs, target_weights, _ = model.get_batch(
              {bucket_id: [(token_ids, [])]}, bucket_id)
          # Get output logits for the sentence.
          state_now, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                               target_weights, bucket_id, True)
          state_now = state_now[0]
          d = cal_distance(state,state_now)
          sen_list.append((sen[0],d))
      sen_list = sorted(sen_list,key = lambda x:x[1])
      print('begin 展示')
      for x in sen_list[:3]:
          print(x)
      '''
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      if data_utils.EOS_ID in outputs:
          outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      sentence_tem = " ".join([rev_fr_vocab[output] for output in outputs])
      print(sentence_tem, end=',')
      print('')
      
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()




def main(_):
    if sys.argv[1] == 'QA':
        decode()#decode()
    else:
        train()

if __name__ == '__main__':
    tf.app.run()
