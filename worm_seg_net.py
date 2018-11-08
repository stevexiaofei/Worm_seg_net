'''
Created on june 12, 2018

author: Edmond
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging
from time import time
import tensorflow as tf

import util
from layers import (weight_variable, weight_variable_devonc, bias_variable,
                            conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax_2,
                            crop_to_shape_v2,cross_entropy)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class SegNet(object):
	def __init__(self,cfg,learning_rate = 0.0017, channels=3, n_class=2,decay_step=2000,decay = 0.96, cost="cross_entropy", 
						cost_kwargs={}, **kwargs):
		print('begin initialize!')
		self.cfg=cfg
		self.n_class = n_class
		self.in_shape =(592,800)
		self.summaries = kwargs.get("summaries", True)
		self.x = tf.placeholder("float", shape=[None, 592, 800, channels])
		self.y = tf.placeholder("float", shape=[None, 592, 800, 1])
		self.Dropout_Rate = tf.placeholder(tf.float32)  # dropout (keep probability)
		self.IsTraining = tf.placeholder(tf.bool)
		self.logits = self._creat_model(self.x,channels,n_class)
		self.loss = -tf.reduce_mean(self.y*tf.log(tf.clip_by_value(self.logits,1e-10,1.0)))+\
				 -tf.reduce_mean((1-self.y)*tf.log(tf.clip_by_value(1-self.logits,1e-10,1.0)))
		self.predicter =tf.sign(self.logits-0.5)
		self.correct_pred = tf.equal(tf.cast(self.predicter, tf.bool), tf.cast(self.y, tf.bool))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
		self.cross_entropy = self.loss
		self.decay_step = decay_step
		self.decay = decay
		self.verification_batch_size =4
		with tf.name_scope('steps'):
			self.train_step = tf.Variable(0, name = 'global_step', trainable= False)
		with tf.name_scope('lr'):
			self.lr = tf.train.exponential_decay(learning_rate, self.train_step, self.decay_step, self.decay, staircase= True, name= 'learning_rate')
		with tf.name_scope('rmsprop'):
			self.rmsprop = tf.train.RMSPropOptimizer(learning_rate= self.lr)
		with tf.name_scope('minimizer'):
			self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(self.update_ops):
				self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)
		self.init = tf.global_variables_initializer()
		tf.summary.scalar('loss', self.loss)
		tf.summary.scalar('cross_entropy', self.cross_entropy)
		tf.summary.scalar('accuracy', self.accuracy)
		tf.summary.scalar('learning_rate', self.lr)
		self.summary_op = tf.summary.merge_all()
		print('end initialize!')
	def _conv(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv'):
		""" Spatial Convolution (CONV2D)
		Args:
			inputs			: Input Tensor (Data Type : NHWC)
			filters		: Number of filters (channels)
			kernel_size	: Size of kernel
			strides		: Stride
			pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
			name			: Name of the block
		Returns:
			conv			: Output Tensor (Convolved Input)
		"""
		with tf.name_scope(name):
			# Kernel for convolution, Xavier Initialisation
			kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
			conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
			return conv 
	def _conv_bn_relu(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv_bn_relu'):
		""" Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
		Args:
			inputs			: Input Tensor (Data Type : NHWC)
			filters		: Number of filters (channels)
			kernel_size	: Size of kernel
			strides		: Stride
			pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
			name			: Name of the block
		Returns:
			norm			: Output Tensor
		"""
		with tf.name_scope(name):
			kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
			conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding='VALID', data_format='NHWC')
			norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.IsTraining)
			#if self.w_summary:
			#	with tf.device('/cpu:0'):
			#		tf.summary.histogram('weights_summary', kernel, collections = ['weight'])
			return norm
	def _conv_block(self, inputs, numOut, name = 'conv_block'):
		""" Convolutional Block
		Args:
			inputs	: Input Tensor
			numOut	: Desired output number of channel
			name	: Name of the block
		Returns:
			conv_3	: Output Tensor
		"""
		with tf.name_scope(name):
			with tf.name_scope('norm_1'):
				norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.IsTraining)
				conv_1 = self._conv(norm_1, int(numOut/2), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
			with tf.name_scope('norm_2'):
				norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.IsTraining)
				pad = tf.pad(norm_2, np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
				conv_2 = self._conv(pad, int(numOut/2), kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
			with tf.name_scope('norm_3'):
				norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.IsTraining)
				conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
		return conv_3
	def _skip_layer(self, inputs, numOut, name = 'skip_layer'):
		""" Skip Layer
		Args:
			inputs	: Input Tensor
			numOut	: Desired output number of channel
			name	: Name of the bloc
		Returns:
			Tensor of shape (None, inputs.height, inputs.width, numOut)
		"""
		with tf.name_scope(name):
			if inputs.get_shape().as_list()[3] == numOut:
				return inputs
			else:
				conv = self._conv(inputs, numOut, kernel_size=1, strides = 1, name = 'conv')
				return conv
	def _residual(self, inputs, numOut, modif = False, name = 'residual_block'):
		""" Residual Unit
		Args:
			inputs	: Input Tensor
			numOut	: Number of Output Features (channels)
			name	: Name of the block
		"""
		with tf.name_scope(name):
			convb = self._conv_block(inputs, numOut)
			skipl = self._skip_layer(inputs, numOut)
			if modif:
				return tf.nn.relu(tf.add_n([convb, skipl], name = 'res_block'))
			else:
				return tf.add_n([convb, skipl], name = 'res_block')
	def _bn_relu(self, inputs):
		norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.IsTraining)
		return norm
	def _pool_layer(self, inputs, numOut, name = 'pool_layer'):
		with tf.name_scope(name):
			bnr_1 = self._bn_relu(inputs)
			pool = tf.contrib.layers.max_pool2d(bnr_1,[2,2],[2,2],padding='VALID')
			pad_1 = tf.pad(pool, np.array([[0,0],[1,1],[1,1],[0,0]]))
			conv_1 = self._conv(pad_1, numOut, kernel_size=3, strides=1, name='conv')
			bnr_2 = self._bn_relu(conv_1)
			pad_2 = tf.pad(bnr_2, np.array([[0,0],[1,1],[1,1],[0,0]]))
			conv_2 = self._conv(pad_2, numOut, kernel_size=3, strides=1, name='conv')
			upsample = tf.image.resize_nearest_neighbor(conv_2, tf.shape(conv_2)[1:3]*2, name = 'upsampling')
		return upsample
	def _attention_iter(self, inputs, lrnSize, itersize, name = 'attention_iter'):
		with tf.name_scope(name):
			numIn = inputs.get_shape().as_list()[3]
			padding = np.floor(lrnSize/2)
			pad = tf.pad(inputs, np.array([[0,0],[1,1],[1,1],[0,0]]))
			U = self._conv(pad, filters=1, kernel_size=3, strides=1)
			pad_2 = tf.pad(U, np.array([[0,0],[padding,padding],[padding,padding],[0,0]]))
			sharedK = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([lrnSize,lrnSize, 1, 1]), name= 'shared_weights')
			Q = []
			C = []
			for i in range(itersize):
				if i ==0:
					conv = tf.nn.conv2d(pad_2, sharedK, [1,1,1,1], padding='VALID', data_format='NHWC')
				else:
					conv = tf.nn.conv2d(Q[i-1], sharedK, [1,1,1,1], padding='SAME', data_format='NHWC')
				C.append(conv)
				Q_tmp = tf.nn.sigmoid(tf.add_n([C[i], U]))
				Q.append(Q_tmp)
			stacks = []
			for i in range(numIn):
				stacks.append(Q[-1]) 
			pfeat = tf.multiply(inputs,tf.concat(stacks, axis = 3) )
		return pfeat
	def _residual_pool(self, inputs, numOut, name = 'residual_pool'):
		with tf.name_scope(name):
			return tf.add_n([self._conv_block(inputs, numOut), self._skip_layer(inputs, numOut), self._pool_layer(inputs, numOut)])
	def _lin(self, inputs, numOut, name = 'lin'):
		l = self._conv(inputs, filters = numOut, kernel_size = 1, strides = 1)
		return self._bn_relu(l)
		
	def _rep_residual(self, inputs, numOut, nRep, name = 'rep_residual'):
		with tf.name_scope(name):
			out = [None]*nRep
			for i in range(nRep):
				if i == 0:
					tmpout = self._residual(inputs,numOut)
				else:
					tmpout = self._residual_pool(out[i-1],numOut)
				out[i] = tmpout
			return out[nRep-1]
	def up_sample(self,inputs,numOut,pool_size = 2,name = 'upsample'):
		with tf.name_scope('upsample'):
			kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([pool_size,pool_size, numOut, inputs.get_shape().as_list()[3]]), name= 'weights')
			#wd = weight_variable_devonc([pool_size, pool_size, numOut// 2, numOut], stddev)
			#bd = bias_variable([features // 2])
			h_deconv = tf.nn.relu(deconv2d(inputs, kernel, pool_size))
		return h_deconv
	def _hg_mcam(self, inputs, n, numOut, nModual, name = 'mcam_hg'):
		with tf.name_scope(name):
			#------------Upper Branch
			pool = tf.contrib.layers.max_pool2d(inputs,[2,2],[2,2],padding='VALID')
			up = []
			low = [] 
			for i in range(nModual):
				if i == 0:
					if n>1:
						tmpup = self._rep_residual(inputs, numOut, n -1)
					else:
						tmpup = self._residual(inputs, numOut)
					tmplow = self._residual(pool, numOut*2)
				else:
					if n>1:
						tmpup = self._rep_residual(up[i-1], numOut, n-1)
					else:
						tmpup = self._residual_pool(up[i-1], numOut)
					tmplow = self._residual(low[i-1], numOut*2)
				up.append(tmpup)
				low.append(tmplow)
				#up[i] = tmpup
				#low[i] = tmplow
			#----------------Lower Branch
			if n>1:
				low2 = self._hg_mcam(low[-1], n-1, numOut*2, nModual)
			else:
				low2 = self._residual(low[-1], numOut*2)
			low3 = self._residual(low2, numOut*2)
			#up_2 = tf.image.resize_nearest_neighbor(low3, tf.shape(low3)[1:3]*2, name = 'upsampling')
			up_2 = self.up_sample(low3,numOut)
			return tf.add_n([up[-1], up_2], name = 'out_hg')
	def _creat_model(self,x,channels, n_class, layers=3, features_root=16, 
				filter_size=3, pool_size=2,summaries=True):
		# Placeholder for the input image
		#nx = tf.shape(x)[1]
		#ny = tf.shape(x)[2]
		#x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
		in_node = x
		#batch_size = tf.shape(x_image)[0]
		with tf.name_scope('model'):
			with tf.name_scope('preprocessing'):
				pad1 = tf.pad(in_node, [[0,0],[1,1],[1,1],[0,0]], name='pad_1')
				conv1 = self._conv_bn_relu(pad1, filters= 8, kernel_size = 3, strides = 1, name = 'conv_channel_to_64')
				in_node = self._residual_pool(conv1, numOut = 16, name = 'r1')
				#in_node = self._skip_layer(in_node,16,name = "conv_channel_64_to_128")
			#pool = tf.contrib.layers.max_pool2d(in_node,[2,2],[2,2],padding='VALID')
			with tf.name_scope('unet'):
				in_node=self._hg_mcam(in_node,3,16,2)
			with tf.name_scope('attention'):
				drop = tf.layers.dropout(in_node, rate=self.Dropout_Rate, training = self.IsTraining)
				output = self._lin(in_node,1)
				#att = self._attention_iter(ll,3,3)
			#upsample = tf.image.resize_nearest_neighbor(att, tf.shape(att)[1:3]*2, name = 'upsampling')
			# with tf.name_scope('output'):
				# out = self._lin(att,2)
		return tf.nn.sigmoid(output)
	def _get_cost(self, logits, cost_name, cost_kwargs):
		"""
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are:
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
		"""
			
		flat_logits = tf.reshape(logits, [-1, self.n_class])
		flat_labels = tf.reshape(self.y, [-1, self.n_class])
		if cost_name == "cross_entropy":
			class_weights = cost_kwargs.pop("class_weights", None)

			if class_weights is not None:
				class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

				weight_map = tf.multiply(flat_labels, class_weights)
				weight_map = tf.reduce_sum(weight_map, axis=1)

				loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                      labels=flat_labels)
				weighted_loss = tf.multiply(loss_map, weight_map)

				loss = tf.reduce_mean(weighted_loss)

			else:
				loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                                 labels=flat_labels))
		elif cost_name == "dice_coefficient":
			eps = 1e-5
			prediction = pixel_wise_softmax_2(logits)
			intersection = tf.reduce_sum(prediction * self.y)
			union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
			loss = -(2 * intersection / (union))

		else:
			raise ValueError("Unknown cost function: " % cost_name)

		return loss
		
	def predict(self, model_path, x_test):
		"""
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
		"""

		init = tf.global_variables_initializer()
		with tf.Session() as sess:
            # Initialize variables
			sess.run(init)

            # Restore model weights from previously saved model
			self.restore(sess, model_path)

			y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
			begin = time()
			for i in range(1000):
				prediction = sess.run(self.predicter, feed_dict={self.x:crop_to_shape_v2(x_test,self.in_shape), 
							 self.Dropout_Rate: 0.,self.IsTraining:False})
			print('time comsumed:',(time()-begin)/1000.)
		return prediction
	def save(self, sess, model_path):
		"""
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
		"""

		saver = tf.train.Saver()
		save_path = saver.save(sess, model_path)
		return save_path
	def restore(self, sess, model_path):
		"""
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
		"""

		saver = tf.train.Saver()
		saver.restore(sess, model_path)
		logging.info("Model restored from file: %s" % model_path)
	def _initialize(self, output_path, prediction_path,restore =False):
		abs_prediction_path = os.path.abspath(prediction_path)
		output_path = os.path.abspath(output_path)
		if not restore:
			logging.info("Removing '{:}'".format(abs_prediction_path))
			shutil.rmtree(abs_prediction_path, ignore_errors=True)
			logging.info("Removing '{:}'".format(output_path))
			shutil.rmtree(output_path, ignore_errors=True)
		else:
			self.restore(sess,output_path)
		if not os.path.exists(abs_prediction_path):
			logging.info("Allocating '{:}'".format(abs_prediction_path))
			os.makedirs(abs_prediction_path)

		if not os.path.exists(output_path):
			logging.info("Allocating '{:}'".format(output_path))
			os.makedirs(output_path)
	
	def train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.25,
							display_step=1,restore=False, write_graph=False, prediction_path='prediction'):
		"""
        Lauches the training process

        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
		"""
		print('begin training')
		save_path = os.path.join(output_path, "model.ckpt")
		if epochs == 0:
			return save_path
		self.prediction_path = prediction_path
		self.model_path = output_path
		self._initialize(output_path,prediction_path)
		#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
		#config=tf.ConfigProto(gpu_options=gpu_options)
		with tf.Session() as sess:
			if write_graph:
				tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

			sess.run(self.init)

			if restore:
				ckpt = tf.train.get_checkpoint_state(output_path)
				if ckpt and ckpt.model_checkpoint_path:
					self.restore(sess, ckpt.model_checkpoint_path)
			#############################
			self.verification_batch_size=2
			self.batch_size=1
			test_x, test_y = data_provider(self.verification_batch_size)
			#print('shape')
			#print(test_x.shape,test_y.shape)
			pred_shape = self.store_prediction(sess, test_x, test_y, "_init")

			summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
			logging.info("Start optimization")

			avg_gradients = None
			seld.save_dict= {'loss':[],'acc':[]}
			for epoch in range(epochs):
				test_x, test_y = data_provider(self.verification_batch_size)
				total_loss = 0
				for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
					batch_x, batch_y = data_provider(self.batch_size)

                    # Run optimization op (backprop)
					_, loss, lr= sess.run(
                        (self.train_rmsprop, self.loss, self.lr),
                        feed_dict={self.x: crop_to_shape_v2(batch_x,self.in_shape),
                                   self.y: crop_to_shape_v2(batch_y,self.in_shape),
										self.IsTraining:True,
                                   self.Dropout_Rate: dropout})

					if step % display_step == 0:
						self.output_minibatch_stats(sess, summary_writer, step, batch_x,batch_y)

					total_loss += loss
				np.savez('log_data.npz',**self.save_dict)
				self.output_epoch_stats(epoch, total_loss, training_iters, lr)
				self.store_prediction(sess, test_x, test_y, "epoch_%s" % epoch)

				save_path = self.save(sess, save_path)
			logging.info("Optimization Finished!")

			return save_path

	def store_prediction(self, sess, batch_x, batch_y, name):
		y = crop_to_shape_v2(batch_y,self.in_shape)
		#print(y.shape)
		prediction = sess.run(self.predicter, feed_dict={self.x: crop_to_shape_v2(batch_x,self.in_shape),
                                                             self.y: y,
                                                             self.Dropout_Rate: 0.,
																		self.IsTraining:False})

		loss = sess.run(self.loss, feed_dict={self.x: crop_to_shape_v2(batch_x,self.in_shape),
                                                  self.y:y ,
														 self.IsTraining:False,
                                                  self.Dropout_Rate: 0})
		
		logging.info("Verification error= {:.1f}%, loss= {:.4f}".format(error_rate(prediction,crop_to_shape_v2(batch_y,self.in_shape)),
                                                                        loss))

		img = util.combine_img_prediction(batch_x, batch_y, prediction)
		util.save_image(img, "%s/%s.jpg" % (self.prediction_path, name))


	def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
		logging.info(
            "Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))

	def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
		# Calculate batch loss and accuracy
		summary_str, loss, acc, predictions = sess.run([self.summary_op,
                                                        self.loss,
                                                        self.accuracy,
                                                        self.predicter],
                                      feed_dict={self.x:crop_to_shape_v2(batch_x,self.in_shape),
                                                      self.y: crop_to_shape_v2(batch_y,self.in_shape),
                                                                  self.IsTraining: False,
																			self.Dropout_Rate:1.})
		self.save_dict['loss'].append(loss)
		self.save_dict['acc'].append(acc)
		summary_writer.add_summary(summary_str, step)
		summary_writer.flush()
		logging.info(
            "Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}".format(step,loss,acc))
def error_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """
    #print(np.unique(predictions),np.unique(labels))
    return 100.0 - (
            100.0 *
            np.sum(np.sign(predictions-0.5) == np.sign(labels-0.5)) /
            (predictions.shape[0] * predictions.shape[1] * predictions.shape[2]))


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V
	
			
			
				