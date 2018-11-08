import cv2
import numpy as np
import os
label_path='D:\\dataset\\worms\\441\\label'
save_path='D:\\dataset\\worms\\441\\labels'
img_path='D:\\dataset\\worms\\441'
worm_img='D:\\dataset\\worms\\image'
worm_label='D:\\dataset\\worms\\label'
label_list=os.listdir(label_path)
dir_name='441'
print((len(label_list)))
new_label_list=[]
for it in label_list:
	new_label_list.append(it.split('-'))

sorted_list = sorted(new_label_list,key=lambda x: int(x[0]))
def plot_contour(img,orgin_image):
	ret,th=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	(_,cnts, _) = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	areas = [cv2.contourArea(cnt) for cnt in cnts]
	max_index = areas.index(max(areas))
	#cv2.drawContours(orgin_image, cnts[max_index], -1, (255,255,255), thickness=1)
	return orgin_image
def processing_io_contour(inner_contour,outer_contour,cur_image,orgin_image):
	img_path_1= os.path.join(label_path,'-'.join(inner_contour))
	img_path_2= os.path.join(label_path,'-'.join(outer_contour))
	print('reading the image....'+img_path_1)
	print('reading the image....'+img_path_2)
	img_1 = cv2.imread(img_path_1,0)
	img_2 = cv2.imread(img_path_2,0)
	orgin_image= plot_contour(img_1,orgin_image)
	orgin_image = plot_contour(img_2,orgin_image)
	#print(np.unique(img_1))
	img_1_reverse = 255-img_1
	contour = cv2.bitwise_and(img_1_reverse,img_2)
	return cv2.bitwise_or(contour,cur_image),orgin_image
def processing_contour(contour,cur_image,orgin_image):
	img_path = os.path.join(label_path,'-'.join(contour))
	print('reading the image....'+img_path)
	img = cv2.imread(img_path,0)
	orgin_image=plot_contour(img,orgin_image)
	#print(img.shape,cur_image.shape)
	return cv2.bitwise_or(img,cur_image), orgin_image
i=0
cur_name=sorted_list[0][0]
cur_image=np.zeros((1200,1600),dtype=np.uint8)
orgin_image = cv2.imread(os.path.join(img_path,cur_name+'.jpg'))
print(len(sorted_list))
while(i<len(sorted_list)):
	print('processing the index:',i)
	if i!=0 and cur_name!=sorted_list[i][0]:
		cv2.imwrite(os.path.join(save_path,dir_name+'.label.'+cur_name+'.jpg'),cur_image)
		cv2.imwrite(os.path.join(save_path,dir_name+'.orgin.'+cur_name+'.jpg'),orgin_image)
		cur_name=sorted_list[i][0]
		cur_image=np.zeros((1200,1600),dtype=np.uint8)
		orgin_image = cv2.imread(os.path.join(img_path,cur_name+'.jpg'))
	if len(sorted_list[i])==3:
		cur_image,orgin_image = processing_io_contour(sorted_list[i],sorted_list[i+1],cur_image,orgin_image)
		i=i+2
	else:
		cur_image,orgin_image = processing_contour(sorted_list[i],cur_image,orgin_image)
		i=i+1
cv2.imwrite(os.path.join(save_path,dir_name+'.label.'+cur_name+'.jpg'),cur_image)
cv2.imwrite(os.path.join(save_path,dir_name+'.orgin.'+cur_name+'.jpg'),orgin_image)
		

