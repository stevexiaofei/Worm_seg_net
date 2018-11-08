import cv2
import numpy as np
import os
def eval_metric(label_path,output_path): 
	label = cv2.imread(label_path,0)
	output = cv2.imread(output_path,0)
	_, label = cv2.threshold(label,0,1,cv2.THRESH_BINARY)
	_, output = cv2.threshold(output,0,1,cv2.THRESH_BINARY)
	label_ = 1- label
	output_ = 1-output
	Dn = label_.sum()#background
	Dp = label.sum()#object
	Qp = cv2.bitwise_and(label,output_).sum()#classify object to background
	Up = cv2.bitwise_and(label_,output).sum()#classify background to object
	OR = Qp/float(Dp) #over segmentation
	UR = Up/float(Dn) #under segmentation
	ER = (Qp+Up)/float(Dp+Dn)
	#print(Qp,Up,Dp,Dn)
	return OR,UR,ER
label_path= 'D:\\dataset\\worms\\label\\441.label.1.jpg'
output_path= 'D:\\dataset\\worms\\label\\441.label.11.jpg'
print(eval_metric(label_path,output_path))