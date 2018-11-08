'''
Created on June 13, 2018

author: Edmond
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import sys
sys.path.append('.')
import image_gen
import worm_seg_net
from image_util import ImageDataProvider
import util
cfg = util.process_config('config.cfg')
if __name__ == '__main__':
	nx = 520
	ny = 696
	training_iters = 100
	epochs = 163
	dropout = 0.25 # Dropout, probability to keep units
	display_step = 2
	restore = True
	
	generator = ImageDataProvider()
	x_test, y_test = generator(4)
	print(x_test.shape,y_test.shape)
	net = worm_seg_net.SegNet(cfg,channels=generator.channels, 
                    n_class=generator.n_class, 
                    cost="cross_entropy")
	net.train(generator,"./unet_trained",training_iters=training_iters, 
                         epochs=epochs,dropout=dropout,display_step=display_step,restore=restore )
	x_test, y_test = generator(1)
	model_path = 'unet_trained\\model.ckpt'
	prediction = net.predict(model_path, x_test)
	print(prediction.shape,util.crop_to_shape_v2(y_test,prediction.shape).shape)
	print("Testing error rate: {:.2f}%".format(unet_v2.error_rate(prediction, util.crop_to_shape_v2(y_test, net.in_shape))))

	