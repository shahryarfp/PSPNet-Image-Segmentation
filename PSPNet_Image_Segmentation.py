#!/usr/bin/env python
# coding: utf-8

# # PSPNet Image Segmentation
# ## Installing mxnet and gluoncv and importing needed Libraries

# In[1]:


get_ipython().system('pip install mxnet')
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
from IPython import display
get_ipython().system('pip install gluoncv')
import gluoncv
ctx = mx.cpu(0) # Using CPU


# ## Loading the Image

# In[2]:


image_path = "PSPNet image.png"
img = image.imread(image_path)
#normalizing image
from gluoncv.data.transforms.presets.segmentation import test_transform
img = test_transform(img, ctx)

display.display(display.Image(image_path, width=1024))


# ## Loading Pre-Trained Model and Predict the Image

# In[10]:


model = gluoncv.model_zoo.get_model('psp_resnet101_ade', pretrained=True)


# In[11]:


# Predicting
output = model.predict(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()


# In[12]:


# Add color to the mask to display the result
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
mask = get_color_pallete(predict, 'pascal_voc')
mask.save('output.png')


# In[13]:


# Displaying the result
display.display(display.Image('output.png', width=1024))


# In[6]:




