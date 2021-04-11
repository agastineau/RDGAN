from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import math_ops

import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
import argparse
import math
import time
import collections
import os
import json

from utils import array2raster

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

parser = argparse.ArgumentParser()
parser.add_argument("--train_tfrecord", help="filename of train_tfrecord", 
default="F:/Anais/Tests_img_wv/tfrecord/train.tfrecords") #change path
parser.add_argument("--test_tfrecord", help="filename of test_tfrecord", default="F:/Anais/Tests_img_wv/tfrecord/test.tfrecords") #change path

parser.add_argument("--mode", required=True, choices=["train","test"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--checkpoint", default=None, help="directory with checkpoints")
parser.add_argument("--max_steps", type=int, help="max training steps")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")

parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=10000, help="write current training images ever display_freq steps")
parser.add_argument("--save_freq", type=int, default=2000, help="save model every save_freq steps")

parser.add_argument("--batch_size",type=int, default=19, help="number of images in batch")

parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--pxs_weight", type=float, default=0.5, help="weight on PXS term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=0.1, help="weight on GAN term for generator gradient")

parser.add_argument("--ndf", type=int, default=32, help="number of generator filters in first conv layer")
parser.add_argument("--train_count", type=int, default=5615,help="number of training data")
parser.add_argument("--test_count", type=int, default=624, help="number of test data")
a=parser.parse_args()

def pxs(output,target):
    
    pixel_dif1_target = target[:, 1:, :, :] - target[:, :-1, :, :]
    pixel_dif2_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    pixel_dif1_output = output[:, 1:, :, :] - output[:, :-1, :, :]
    pixel_dif2_output = output[:, :, 1:, :] - output[:, :, :-1, :]

    prod_scalaire = math_ops.abs(-pixel_dif2_target[:, :-1, :, :]*pixel_dif1_output[:, :, :-1, :] + 
                                  pixel_dif1_target[:, :, :-1, :]*pixel_dif2_output[:, :-1, :, :])

    return tf.reduce_mean(prod_scalaire)

def write_file_txt(fichier,valeur):
    if fichier == "discrim_loss":
        mon_fichier = open(a.output_dir + "/discrim_loss.txt","a")
    if fichier == "gen_loss_GAN":
        mon_fichier = open(a.output_dir + "/gen_loss_GAN.txt","a")
    if fichier == "gen_loss_L1":
        mon_fichier = open(a.output_dir + "/gen_loss_l1.txt","a")
    if fichier == "gen_loss_pxs":
        mon_fichier = open(a.output_dir + "/gen_loss_pxs.txt","a")
        
    mon_fichier.write(str(valeur)+"\n")
    mon_fichier.close()
    

EPS = 1e-12
Examples = collections.namedtuple("Examples", "inputs1, inputs2, targets, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_loss_pxs, gen_grads_and_vars, train")

def conv(batch_input, kernel_size, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [kernel_size, kernel_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(batch_input, filter, [1,stride, stride, 1], padding='SAME')
        return conv

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x=tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def create_generator(generator_inputs1, generator_inputs2, generator_outputs_channels):
    
    num_spectral = 4
    num_fm = 32
    num_res = 4
    weight_decay = 1e-5
    
    layers = tf.concat([generator_inputs1,generator_inputs2],3)

    rs_cat1 = ly.conv2d(layers,num_outputs = num_fm,kernel_size = 3,stride = 1,
                       weights_regularizer = ly.l2_regularizer(weight_decay),
                       weights_initializer = ly.variance_scaling_initializer(),
                       activation_fn = tf.nn.relu)
    
    #block 1   
    rs1 = ly.conv2d(rs_cat1,num_outputs = num_fm,kernel_size = 3,stride = 1,
                        weights_regularizer = ly.l2_regularizer(weight_decay),
                        weights_initializer = ly.variance_scaling_initializer(),
                        activation_fn = tf.nn.relu)

    rs_cat2 = tf.concat([rs_cat1,rs1],3)

    rs1 = ly.conv2d(rs_cat2,num_outputs = 64,kernel_size = 3,stride = 1,
                        weights_regularizer = ly.l2_regularizer(weight_decay),
                        weights_initializer = ly.variance_scaling_initializer(),
                        activation_fn = tf.nn.relu)

    rs_cat3 = tf.concat([rs_cat2,rs1],3)
    
    rs1 = ly.conv2d(rs_cat3,num_outputs = 128,kernel_size = 3,stride = 1,
                        weights_regularizer = ly.l2_regularizer(weight_decay),
                        weights_initializer = ly.variance_scaling_initializer(),
                        activation_fn = tf.nn.relu)

    rs_cat4 = tf.concat([rs_cat3,rs1],3)

    rs1 = ly.conv2d(rs_cat4,num_outputs = num_fm,kernel_size = 1,stride = 1, 
                   weights_regularizer = ly.l2_regularizer(weight_decay), 
                   weights_initializer = ly.variance_scaling_initializer(),
                   activation_fn = None)

    rs1 = tf.add(rs1,rs_cat1);

    #block 2
    rs2 = ly.conv2d(rs1,num_outputs = num_fm,kernel_size = 3,stride = 1,
                        weights_regularizer = ly.l2_regularizer(weight_decay),
                        weights_initializer = ly.variance_scaling_initializer(),
                        activation_fn = tf.nn.relu)

    rs2_cat2 = tf.concat([rs1,rs2],3)

    rs2 = ly.conv2d(rs2_cat2,num_outputs = 64,kernel_size = 3,stride = 1,
                        weights_regularizer = ly.l2_regularizer(weight_decay),
                        weights_initializer = ly.variance_scaling_initializer(),
                        activation_fn = tf.nn.relu)
    
    rs2_cat3 = tf.concat([rs2_cat2,rs2],3)

    rs2 = ly.conv2d(rs2_cat3,num_outputs = 128,kernel_size = 3,stride = 1,
                        weights_regularizer = ly.l2_regularizer(weight_decay),
                        weights_initializer = ly.variance_scaling_initializer(),
                        activation_fn = tf.nn.relu)

    rs2_cat4 = tf.concat([rs2_cat3,rs2],3)

    rs2 = ly.conv2d(rs2_cat4,num_outputs = num_fm,kernel_size = 1,stride = 1, 
                   weights_regularizer = ly.l2_regularizer(weight_decay), 
                   weights_initializer = ly.variance_scaling_initializer(),
                   activation_fn = None)

    rs2 = tf.add(rs2,rs1);

    #block 3
    rs3 = ly.conv2d(rs2,num_outputs = num_fm,kernel_size = 3,stride = 1,
                        weights_regularizer = ly.l2_regularizer(weight_decay),
                        weights_initializer = ly.variance_scaling_initializer(),
                        activation_fn = tf.nn.relu)

    rs3_cat2 = tf.concat([rs2,rs3],3)

    rs3 = ly.conv2d(rs3_cat2,num_outputs = 64,kernel_size = 3,stride = 1,
                        weights_regularizer = ly.l2_regularizer(weight_decay),
                        weights_initializer = ly.variance_scaling_initializer(),
                        activation_fn = tf.nn.relu)

    rs3_cat3 = tf.concat([rs3_cat2,rs3],3)

    rs3 = ly.conv2d(rs3_cat3,num_outputs = 128,kernel_size = 3,stride = 1,
                        weights_regularizer = ly.l2_regularizer(weight_decay),
                        weights_initializer = ly.variance_scaling_initializer(),
                        activation_fn = tf.nn.relu)
    
    rs3_cat4 = tf.concat([rs3_cat3,rs3],3)

    rs3 = ly.conv2d(rs3_cat4,num_outputs = num_fm,kernel_size = 1,stride = 1, 
                   weights_regularizer = ly.l2_regularizer(weight_decay), 
                   weights_initializer = ly.variance_scaling_initializer(),
                   activation_fn = None)

    rs3 = tf.add(rs3,rs2);

    #block 4
    rs4 = ly.conv2d(rs3,num_outputs = num_fm,kernel_size = 3,stride = 1,
                        weights_regularizer = ly.l2_regularizer(weight_decay),
                        weights_initializer = ly.variance_scaling_initializer(),
                        activation_fn = tf.nn.relu)
    rs4_cat2 = tf.concat([rs3,rs4],3)

    rs4 = ly.conv2d(rs4_cat2,num_outputs = 64,kernel_size = 3,stride = 1,
                        weights_regularizer = ly.l2_regularizer(weight_decay),
                        weights_initializer = ly.variance_scaling_initializer(),
                        activation_fn = tf.nn.relu) 
    rs4_cat3 = tf.concat([rs4_cat2,rs4],3)

    rs4 = ly.conv2d(rs4_cat3,num_outputs = 128,kernel_size = 3,stride = 1,
                        weights_regularizer = ly.l2_regularizer(weight_decay),
                        weights_initializer = ly.variance_scaling_initializer(),
                        activation_fn = tf.nn.relu)
    rs4_cat4 = tf.concat([rs4_cat3,rs4],3)
    
    rs4 = ly.conv2d(rs4_cat4,num_outputs = num_fm,kernel_size = 1,stride = 1, 
                   weights_regularizer = ly.l2_regularizer(weight_decay), 
                   weights_initializer = ly.variance_scaling_initializer(),
                   activation_fn = None)
    
    rs4 = tf.add(rs4,rs3);

    #final convolution
    rs_final = ly.conv2d(rs4,num_outputs = num_spectral,kernel_size = 1,stride = 1, 
                   weights_regularizer = ly.l2_regularizer(weight_decay), 
                   weights_initializer = ly.variance_scaling_initializer(),
                   activation_fn = None)

    return rs_final

def create_model(inputs1, inputs2, targets):
    def create_discriminator(discrim_inputs, discrim_targets):      
        n_layers = 3
        layers = []

        input = tf.concat([discrim_inputs, discrim_targets], 3)

        with tf.variable_scope("layer_1"):
            convolved = conv(input, 3, a.ndf, 2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer has stride 1
                convolved = conv(layers[-1], 3, out_channels, stride=stride)
                rectified = lrelu(convolved, 0.2)
                layers.append(rectified)

        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, 3, 1, 1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs2 = create_generator(inputs1, inputs2, out_channels)
        outputs = tf.add(outputs2,inputs1)

    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            predict_real = create_discriminator(inputs1, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = create_discriminator(inputs1, outputs)

    with tf.name_scope("discriminator_loss"):
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake+EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss_pxs = pxs(outputs,targets)
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight + gen_loss_pxs * a.pxs_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1, gen_loss_pxs])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real = predict_real,
        predict_fake = predict_fake,
        discrim_loss = ema.average(discrim_loss),
        discrim_grads_and_vars = discrim_grads_and_vars,
        gen_loss_GAN = ema.average(gen_loss_GAN),
        gen_loss_L1 = ema.average(gen_loss_L1),
        gen_loss_pxs = ema.average(gen_loss_pxs),
        gen_grads_and_vars = gen_grads_and_vars,
        outputs= outputs,
        train = tf.group(update_losses, incr_global_step, gen_train),
    )

def load_examples():
    if a.mode == 'train':
        filename_queue = tf.train.string_input_producer([a.train_tfrecord])
    elif a.mode =='test':
        filename_queue = tf.train.string_input_producer([a.test_tfrecord])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'im_mul_raw': tf.FixedLenFeature([], tf.string),
                                           'im_blur_raw': tf.FixedLenFeature([], tf.string),
                                           'im_pan_raw': tf.FixedLenFeature([], tf.string)
                                       })

    im_mul_raw = tf.decode_raw(features['im_mul_raw'], tf.uint8)
    im_mul_raw = tf.reshape(im_mul_raw, [128, 128, 4])
    im_mul_raw = tf.cast(im_mul_raw,tf.float32)
    im_blur_raw = tf.decode_raw(features['im_blur_raw'], tf.uint8)
    im_blur_raw = tf.reshape(im_blur_raw, [128, 128, 4])
    im_blur_raw = tf.cast(im_blur_raw, tf.float32)
    im_pan_raw = tf.decode_raw(features['im_pan_raw'], tf.uint8)
    im_pan_raw = tf.reshape(im_pan_raw, [128, 128, 1])
    im_pan_raw = tf.cast(im_pan_raw, tf.float32)
    if a.mode == 'train':
        inputs1_batch, inputs2_batch, targets_batch = tf.train.shuffle_batch([im_blur_raw, im_pan_raw, im_mul_raw],
                                              batch_size=a.batch_size, capacity=200,
                                              min_after_dequeue=100)
        steps_per_epoch = int(a.train_count / a.batch_size)
    elif a.mode =='test':
        inputs1_batch, inputs2_batch, targets_batch = tf.train.batch([im_blur_raw, im_pan_raw, im_mul_raw],
                                              batch_size=a.batch_size, capacity=200)
        steps_per_epoch = int(a.test_count / a.batch_size)

    return Examples(
        inputs1=inputs1_batch,
        inputs2=inputs2_batch,
        targets=targets_batch,
        steps_per_epoch=steps_per_epoch,
    )

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for i in range((fetches["inputs1"].shape[0])):
        name = '%d'%i
        for kind in ["inputs1","inputs2", "outputs", "targets"]:
            if a.mode == "train":
                filename = "train-"+ name + "-" + kind + ".tif"
                if step is not None:
                    filename = "%08d-%s" % (step, filename)
            else:
                name = '%d'%(i+a.batch_size*step)
                filename = "test-" + name + "-" + kind +".tif"

            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            if kind is not "inputs2":
                array2raster(out_path, [0,0], 128, 128, contents.transpose(2,0,1), 4)
            else:
                array2raster(out_path, [0, 0], 128, 128, contents.reshape((128,128)), 1)

def main():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

    for k,v in a._get_kwargs():
        print (k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples()
    model = create_model(examples.inputs1, examples.inputs2, examples.targets)

    with tf.name_scope("images"):
        display_fetches = {
            "inputs1": examples.inputs1,
            "inputs2": examples.inputs2,
            "targets": examples.targets,
            "outputs": model.outputs,
        }
    with tf.name_scope("inputs1_summary"):
        tf.summary.image("inputs1", examples.inputs1)

    with tf.name_scope("inputs2_summary"):
        tf.summary.image("inputs2", examples.inputs2)

    with tf.name_scope("targets1_summary"):
        tf.summary.image("targets1", examples.targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", model.outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", model.predict_real)

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", model.predict_fake)

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    tf.summary.scalar("generator_loss_pxs", model.gen_loss_pxs)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq >0 ) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    with sv.managed_session(config=tf.ConfigProto(log_device_placement=True))  as sess:
        print("parameter_count = ", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            max_steps = int(a.test_count/a.batch_size)
            for i in range(max_steps):
                results = sess.run(display_fetches)
                save_images(results, i)
        else:
            start = time.time()
            
            for step  in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1
                    fetches["gen_loss_pxs"] = model.gen_loss_pxs

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    save_images(results["display"], step=results["global_step"])

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    write_file_txt("discrim_loss",results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    write_file_txt("gen_loss_GAN",results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])
                    write_file_txt("gen_loss_L1",results["gen_loss_L1"])
                    print("gen_loss_pxs", results["gen_loss_pxs"])
                    write_file_txt("gen_loss_pxs",results["gen_loss_pxs"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break
                    

main()



