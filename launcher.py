'''
Created on June 13, 2018

author: Edmond
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import sys
sys.path.append('.')
import image_gen
import worm_seg_net
import util
if __name__ == '__main__':
	nx = 520
	ny = 696
	training_iters = 25
	epochs = 30
	dropout = 0.25 # Dropout, probability to keep units
	display_step = 2
	restore = False
	
	generator = image_gen.WormDataProvider(nx, ny, 
		LabelPath='D:\\dataset\\deepworm\\BBBC010_v1_foreground\\BBBC010_v1_foreground', 
		ImagePath='D:\\dataset\\deepworm\\BBBC010_v1_images\\BBBC010_v1_images')
	x_test, y_test = generator(4)
	print(x_test.shape,y_test.shape)
	net = worm_seg_net.SegNet(channels=generator.channels, 
                    n_class=generator.n_class, 
                    cost="cross_entropy")
	net.train(generator,"./unet_trained",training_iters=training_iters, 
                         epochs=epochs,dropout=dropout,display_step=display_step,restore=restore )
	x_test, y_test = generator(1)
	model_path = 'model_trained\\model.ckpt'
	prediction = net.predict(model_path, x_test)
	print(prediction.shape,util.crop_to_shape_v2(y_test,prediction.shape).shape)
	print("Testing error rate: {:.2f}%".format(unet_v2.error_rate(prediction, util.crop_to_shape_v2(y_test, net.in_shape))))

	