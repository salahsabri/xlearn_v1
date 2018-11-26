from xlearn.segmentation import seg_train, seg_predict
from skimage import io
import imageio
import os
import numpy as np
#import dxchange
n=1

with open('paths.txt') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

batch_size = 4000
nb_epoch = 50
nb_down = 3
nb_gpu = 1

# define the data path
spath =content[0]

# define the path to save the training weights
wpath = content[2]




#read the training input 
image_collection=io.imread_collection(os.path.join(content[0],'*.tif'))
dat=io.concatenate_images(image_collection)
number_slice, height,width=dat.shape
A=[]
for i in range(n):
    A.append(dat[i])
imgx=np.concatenate(A,axis=0)

#read the training output 
image_collection2=io.imread_collection(os.path.join(content[1],'*.tif'))


dat2=io.concatenate_images(image_collection2)
number_slice2,height2,width2=dat2.shape
A2=[]
for i in range(n):
    A2.append(dat2[i])
imgy=np.concatenate(A2,axis=0)


# train the model
mdl = seg_train(imgx, imgy, batch_size = batch_size, nb_epoch = nb_epoch, nb_down = nb_down, nb_gpu = nb_gpu)

# save the trained weights
mdl.save_weights(wpath)



# segmentation for the testing data
test=seg_predict(imgx[1], wpath, spath, nb_down = nb_down, nb_gpu = nb_gpu)
imageio.imsave(os.path.join(content[3],os.listdir(content[0])[n]),test)
