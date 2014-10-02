import vigra
import opengm
import numpy
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pylab
import matplotlib.cm as cm
import Image
from matplotlib.colors import LinearSegmentedColormap


img = vigra.readImage('/home/tbeier/datasets/BSR/BSDS500/data/images/train/56028.jpg')
#img = vigra.readImage('/home/tbeier/datasets/BSR/BSDS500/data/images/train/118035.jpg')
img = vigra.readImage('/home/tbeier/datasets/BSR/BSDS500/data/images/val/21077.jpg')
#img = vigra.readImage('/home/tbeier/datasets/BSR/BSDS500/data/images/val/101087.jpg')
#img = vigra.resize(img, (img.shape[0]/1, img.shape[1]/1))#[::1,::1,:]
imgLab = vigra.colors.transform_RGB2Lab(img)


scalesForStrength = (0.4, 0.8)
scalesForOrientation = (0.1, 0.1)
gamma = 0.002
offset = -0.5

structTensorForStrength = vigra.filters.structureTensor(imgLab,*scalesForStrength)
structTensorOrientation = vigra.filters.structureTensor(imgLab,*scalesForOrientation)
eigenReprForStrength  = vigra.filters.tensorEigenRepresentation2D(structTensorForStrength)
eigenReprForOrientation  = vigra.filters.tensorEigenRepresentation2D(structTensorOrientation)


edgeIndicator = eigenReprForStrength[:, :, 0]
orientation = eigenReprForOrientation[:, :, 2]
edgeWeight = numpy.exp(-1.0*gamma*edgeIndicator) + offset
#edgeWeight[0,0] = -1.0
#edgeWeight[0,1] = 1.0







matplotlib.cm.ScalarMappable()

cdict = {'red':   [(0.0,  1.0, 1.0),
                   (0.33, 0.0, 0.0),
                   (0.66, 0.0, 0.0),
                   (1.0,  1.0, 1.0)],

         'green': [(0.0,  0.0, 0.0),
                   (0.33, 1.0, 1.0),
                   (0.66, 0.0, 0.0),
                   (1.0,  0.0, 0.0)],

         'blue':  [(0.0,  0.0, 0.0),
                   (0.33, 0.0, 0.0),
                   (0.66, 1.0, 1.0),
                   (1.0,  0.0, 0.0)]
        }

periodicMap = LinearSegmentedColormap('BlueRed1', cdict)


mapper = matplotlib.cm.ScalarMappable(cmap=periodicMap)
mapper.set_array(orientation)
mapper.autoscale()
procecced = mapper.to_rgba(orientation)
f = pylab.figure()

irelevantOrientations = numpy.where(edgeWeight>0)
proceccedC = procecced[:, :, 0:3]
proceccedC[irelevantOrientations] = 0
print procecced.shape

images = [img/255.0, edgeIndicator, (procecced), edgeWeight]
cmaps   = [None, 'gray', periodicMap, 'seismic']
for n, (dimg, cmap) in enumerate( zip(images,cmaps) ):
    f.add_subplot(2, 2, n)  # this line outputs images on top of each other
    f.add_subplot(2, 2, n)  # this line outputs images side-by-side
    pylab.imshow(dimg.view(numpy.ndarray).swapaxes(0,1),cmap=cmap)
    pylab.colorbar()
pylab.title('Double image')
pylab.show()
