import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from skfuzzy.cluster import cmeans
def loadData(filePath):
	 f = open(filePath,'rb') 
	 data = [] 
	 img = image.open(f) 
	 m,n = img.size 
	 for i in range(m): 
		for j in range(n): 
			x,y,z = img.getpixel((i,j)) 
			data.append([x/256.0,y/256.0,z/256.0]) 
			f.close()
	 return np.mat(data),m,n
imgData,row,col = loadData('11.jpg') 
imgData = imgData.T
center, u, u0, d, jm, p, fpc = cmeans(imgData, m=2, c=2, error=0.0001, maxiter=1000)
for i in u:
	label = np.argmax(u, axis=0)
label = label.reshape([row,col])
pic_new = image.new("L", (row, col))
for i in range(row): 
	for j in range(col): 
		pic_new.putpixel((i,j), int(256/(label[i][j]+1)))
pic_new.save("result-bull-5.jpg", "JPEG")