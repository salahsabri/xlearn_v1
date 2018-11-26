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
imgx=imageio.imread(os.path.join(content[0],os.listdir(content[0])[0]))

#read the training output 
imgy=imageio.imread(os.path.join(content[1],os.listdir(content[1])[0]))


# train the model
mdl = seg_train(imgx, imgy, batch_size = batch_size, nb_epoch = nb_epoch, nb_down = nb_down, nb_gpu = nb_gpu)

# save the trained weights
mdl.save_weights(wpath)


#read the test input 
test_x=imageio.imread(os.path.join(content[0],os.listdir(content[0])[2]))


# segmentation for the testing data
test=seg_predict(test_x, wpath, spath, nb_down = nb_down, nb_gpu = nb_gpu)
imageio.imsave(os.path.join(content[3],os.listdir(content[0])[n]),test)
