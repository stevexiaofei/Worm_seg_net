# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Toy example, generates images at random that can be used for training

Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import SimpleITK as sitk
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import cv2
import os
import scipy.misc
from random import randint
from image_util import BaseDataProvider

class GrayScaleDataProvider(BaseDataProvider):
    channels = 1
    n_class = 2
    
    def __init__(self, nx, ny, **kwargs):
        super(GrayScaleDataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        rect = kwargs.get("rectangles", False)
        if rect:
            self.n_class=3
        
    def _next_data(self):
        return create_image_and_label(self.nx, self.ny, **self.kwargs)
class SingleOutProvider(BaseDataProvider):
	channels = 1
	n_class = 2
	def __init__(self, nx, ny, **kwargs):
		super(SingleOutProvider, self).__init__()
		self.nx = nx
		self.ny = ny
		self.kwargs= kwargs
		
		path= "D:\\dataset\\deepworm\\BBBC010_v1_foreground_eachworm\\BBBC010_v1_foreground_eachworm"
		files =os.listdir(path)
		f_name = lambda f:os.path.join(path,f)
		files=files[1:]
		contours=[]
		rects=[]
		for i,it in enumerate(files):
			img=cv2.imread(f_name(it),0)
			(_,cnts, hier) = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
			if len(cnts)==1:
				(x, y, w, h) = cv2.boundingRect(cnts[0])
				contours.append(np.squeeze(cnts[0], axis=1))
				rects.append((x, y, w, h,x+w/2,y+h/2))
		rects = np.array(rects)
		self.contours=contours
		self.rects=rects
		self.num=len(rects)
	def transfer_loc(self,contour,rects,angle,scale=1.0,center=None):
		x,y,w,h,cx,cy = rects
		center =(cx,cy) if center is None else center
		angle =angle/180.*np.pi
		contour = contour-np.array([cx,cy])
		rotate = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
		contour = np.dot(rotate,contour.T).T*scale
		contour +=np.array(center)
        #todo maybe add a filter to filter the points that are out of bound
		return contour
	def generate_inner_points(self,cnt):
		(x,y,w,h)=cv2.boundingRect(cnt)
		x_list=(x+w*np.random.random((10,))).astype(np.int)
		cnt_points=filter(lambda x: x[0]>x_list[0]-2 and x[0]<x_list[0]+2,cnt)
		cnt_points=list(cnt_points)
		y_loc_ =[p[1] for p in cnt_points]
		y_max=max(y_loc_)
		y_min=min(y_loc_)
		h=y_max-y_min
		y_list=(y_min+h*np.random.random((10,))).astype(np.int)
		x_list=np.full(x_list.shape,x_list[0])
        #print('x_list',x_list)
		points=np.array([x_list,y_list]).T
        #print(x_list[0])
        #print(points)
        #print((x,y,w,h))
        #res =[cv2.pointPolygonTest(cnt,tuple(point),False) for point in points]
		res_ =[cv2.pointPolygonTest(cnt,tuple(point),True) for point in points]
		index =res_.index(max(res_))
        #print('index',index)
		return points[index]
	def add_other_worms(self,mask,cnt,rect,center_point):
		x,y,w,h,cx,cy = rect
		cnt=cnt-np.array([cx,cy],dtype=np.int32)
		center=np.random.random((2,))*256
		cnt_=cnt+np.array([center[0],center[1]],dtype=np.int32)

		while cv2.pointPolygonTest(cnt_,center_point,False)==1:
			center=(np.random.random((2,))*256).astype(np.uint32)
			cnt_=cnt+np.array([center[0],center[1]])
		cv2.drawContours(mask, [cnt_], -1, 255, thickness=-1)
        #mask=cv2.cv2.polylines(mask,[cnt_],True,[255,0,0],1)
		return mask
	def _next_data(self):
		center_index= randint(0,self.num-1)
		random_angle = np.random.random()*360
		cnt = self.transfer_loc(self.contours[center_index],self.rects[center_index],random_angle,scale=1.0).astype(np.int32)
		p= self.generate_inner_points(cnt)
		cnt=cnt-p
		cnt+=np.array([128,128])
		mask_=cv2.drawContours(np.zeros((256,256),dtype=np.uint8), [cnt], -1,255, thickness=-1)
		label= mask_.copy()
		for i in range(3):
			random_index= randint(0,self.num-1)
			mask_=self.add_other_worms(mask_, self.contours[random_index], self.rects[random_index], (128,128))
		return mask_, label
class ParticleDateProvider():
	channels = 1
	n_class =2
	def __init__(self,nx,ny,R,**kwargs):
		self.nx = nx
		self.ny = ny
		self.R = R
		self.kwargs = kwargs
	def _load_data_and_label(self,r_min = 23, r_max = 27, border = 80,sigma = 400):
		nx=self.nx
		ny=self.ny
		image = np.zeros((nx, ny, 1))
		a = np.random.randint(border, nx-border)
		b = np.random.randint(border, ny-border)
		r = np.random.randint(r_min, r_max)
		h = np.random.randint(1,255)
		y,x = np.ogrid[-a:nx-a, -b:ny-b]
		m = x*x + y*y <= r*r
		image[m] =255.
		image += np.random.normal(scale=sigma, size=image.shape)/22.
		image -= np.amin(image)
		image /= np.amax(image)
		label = r<self.R
		return image,label
		
	def __call__(self, n):
		image_batch=[]
		label_batch=[]
		for i in range(n):
			img,label =self._load_data_and_label()
			image_batch.append(img)
			label_batch.append(label)
		image_batch = np.stack(image_batch)
		label_batch = np.stack(label_batch).astype(np.uint8)
		one = OneHotEncoder(n_values=2)
		label_batch = one.fit_transform(label_batch.reshape(-1,1)).toarray()
		return image_batch,label_batch
		
class WormDataProvider(BaseDataProvider):
	channels = 1
	n_class = 2
	
	def __init__(self,nx,ny,**kwargs):
		super(WormDataProvider,self).__init__()
		self.nx = nx
		self.ny = ny
		self.kwargs = kwargs
		self.file_idx = -1
		self.LabelPath = kwargs.get("LabelPath", None)
		assert(self.LabelPath is not None)
		self.ImagePath = kwargs.get('ImagePath', None)
		assert(self.ImagePath is not None)
		files = os.listdir(self.ImagePath)
		self.data_files = list(filter(lambda x: '_w2_' in x,files))
		print("Number of files used: %s" % len(self.data_files))
	def _cylce_file(self):
		self.file_idx += 1
		if self.file_idx >= len(self.data_files):
			self.file_idx = 0 
			np.random.shuffle(self.data_files)
	def _load_file(self, path, dtype=np.float32):
		#return np.array(Image.open(path), dtype)
		return cv2.imread(path).astype(dtype)
	def ReadImage(self,path):
		img3 = sitk.ReadImage(path)
		cvImg = sitk.GetArrayFromImage(img3)
		cvImg = cvImg-cvImg.min()
		cvImg = cvImg.astype(np.float)/cvImg.max()*255.
		return cvImg.astype(np.uint8)
	def _next_data(self):
		self._cylce_file()
		image_name = self.data_files[self.file_idx]
		label_name = image_name[33:36]+'_binary.png' 
		img = self.ReadImage(os.path.join(self.ImagePath,image_name)).astype(np.float32)
		label = cv2.imread(os.path.join(self.LabelPath,label_name), cv2.IMREAD_GRAYSCALE).astype(bool)
		return img,label
		
class RgbDataProvider(BaseDataProvider):
    channels = 3
    n_class = 2
    
    def __init__(self, nx, ny, **kwargs):
        super(RgbDataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        rect = kwargs.get("rectangles", False)
        if rect:
            self.n_class=3

        
    def _next_data(self):
        data, label = create_image_and_label(self.nx, self.ny, **self.kwargs)
        return to_rgb(data), label

def create_image_and_label(nx,ny, cnt = 10, r_min = 5, r_max = 50, border = 92, sigma = 20, rectangles=False):
    
    
    image = np.ones((nx, ny, 1))
    label = np.zeros((nx, ny, 3), dtype=np.bool)
    mask = np.zeros((nx, ny), dtype=np.bool)
    for _ in range(cnt):
        a = np.random.randint(border, nx-border)
        b = np.random.randint(border, ny-border)
        r = np.random.randint(r_min, r_max)
        h = np.random.randint(1,255)

        y,x = np.ogrid[-a:nx-a, -b:ny-b]
        m = x*x + y*y <= r*r
        mask = np.logical_or(mask, m)

        image[m] = h

    label[mask, 1] = 1
    
    if rectangles:
        mask = np.zeros((nx, ny), dtype=np.bool)
        for _ in range(cnt//2):
            a = np.random.randint(nx)
            b = np.random.randint(ny)
            r =  np.random.randint(r_min, r_max)
            h = np.random.randint(1,255)
    
            m = np.zeros((nx, ny), dtype=np.bool)
            m[a:a+r, b:b+r] = True
            mask = np.logical_or(mask, m)
            image[m] = h
            
        label[mask, 2] = 1
        
        label[..., 0] = ~(np.logical_or(label[...,1], label[...,2]))
    
    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)
    
    if rectangles:
        return image, label
    else:
        return image, label[..., 1]




def to_rgb(img):
    img = img.reshape(img.shape[0], img.shape[1])
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    blue = np.clip(4*(0.75-img), 0, 1)
    red  = np.clip(4*(img-0.25), 0, 1)
    green= np.clip(44*np.fabs(img-0.5)-1., 0, 1)
    rgb = np.stack((red, green, blue), axis=2)
    return rgb

