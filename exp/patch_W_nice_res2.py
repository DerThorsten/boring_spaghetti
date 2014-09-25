import vigra
import opengm
import numpy
import matplotlib.pyplot as plt
img = vigra.readImage('/home/tbeier/datasets/BSR/BSDS500/data/images/train/56028.jpg')
img = vigra.readImage('/home/tbeier/datasets/BSR/BSDS500/data/images/train/118035.jpg')
img = img[::1, ::1,:]
grad = vigra.filters.gaussianGradientMagnitude(vigra.colors.transform_Lab2RGB(img), 1.5).squeeze()
grad -= grad.min()
grad /= grad.max()
grad2 = grad.copy()
grad2[numpy.where(grad2<0.3)] = 0

grad2 = numpy.exp(1.5*grad2)-1.0
show = True

if show:
    imgplot = plt.imshow(grad2.swapaxes(0,1))
    plt.colorbar()
    plt.show()

expGrad = numpy.exp(-2.1*grad)
w =  2*expGrad -1.0
w-=w.min()

if show:
    imgplot = plt.imshow(w.swapaxes(0,1))
    plt.colorbar()
    plt.show()

gm = opengm.adder.gridPatchAffinityGm(grad2.astype(numpy.float64), 10.0*w.astype(numpy.float64), 40, 5 ,20, 0.01)
print gm

verbose = True
useQpbo = False
useCgc = False
useWs = False

with opengm.Timer("with new method"):

    fusionParam = opengm.InfParam(fusionSolver = 'cgc', planar=False)
    arg = None
    if useQpbo:
        infParam = opengm.InfParam(
            numStopIt=0,
            numIt=200,
            generator='qpboBased',
            fusionParam = fusionParam
        )
        inf=opengm.inference.IntersectionBased(gm, parameter=infParam)
        # inf.setStartingPoint(arg)
        # start inference (in this case verbose infernce)
        visitor=inf.verboseVisitor(printNth=1,multiline=False)
        if verbose:
            inf.infer(visitor)
        else:
            inf.infer()
        inf.infer()
        arg = inf.arg()


    proposalParam = opengm.InfParam(
        randomizer = opengm.weightRandomizer(noiseType='normalAdd',noiseParam=1.700000001, ignoreSeed=False),
        stopWeight=0.0,
        reduction=0.999,
        setCutToZero=False
    )
    


    infParam = opengm.InfParam(
        numStopIt=20,
        numIt=100,
        generator='randomizedHierarchicalClustering',
        proposalParam=proposalParam,
        fusionParam = fusionParam
    )


    inf=opengm.inference.IntersectionBased(gm, parameter=infParam)
    if arg is not None:
        inf.setStartingPoint(arg)
    # start inference (in this case verbose infernce)
    visitor=inf.verboseVisitor(printNth=1,multiline=False)
    if verbose:
        inf.infer(visitor)
    else:
        inf.infer()
    arg = inf.arg()

    if useWs:
        print "ws"
        proposalParam = opengm.InfParam(
            randomizer = opengm.weightRandomizer(noiseType='normalAdd',noiseParam=1.100000001,ignoreSeed=False),
            seedFraction = 0.005
        )
        infParam = opengm.InfParam(
            numStopIt=20,
            numIt=10,
            generator='randomizedWatershed',
            proposalParam=proposalParam,
            fusionParam = fusionParam
        )


        inf=opengm.inference.IntersectionBased(gm, parameter=infParam)
        if arg is not None:
            inf.setStartingPoint(arg)
        # start inference (in this case verbose infernce)
        visitor=inf.verboseVisitor(printNth=1,multiline=False)
        if verbose:
            inf.infer(visitor)
        else:
            inf.infer()
        arg = inf.arg()



    if useQpbo:
        infParam = opengm.InfParam(
            numStopIt=0,
            numIt=40,
            generator='qpboBased',
            fusionParam = fusionParam
        )
        inf=opengm.inference.IntersectionBased(gm, parameter=infParam)
        inf.setStartingPoint(arg)
        # start inference (in this case verbose infernce)
        visitor=inf.verboseVisitor(printNth=10)

    if useCgc:
        print "cgc"

        infParam = opengm.InfParam(
            planar=False,
            startFromThreshold=False,
            doCutMove = False,
            doGlueCutMove = True,
            maxIterations = 1
        )
        inf=opengm.inference.Cgc(gm, parameter=infParam)
        if arg is not None:
            inf.setStartingPoint(arg)
        # start inference (in this case verbose infernce)
        visitor=inf.verboseVisitor(printNth=10)
        if verbose:
            inf.infer(visitor)
        else:
            inf.infer()

        arg = inf.arg()


    print gm.evaluate(arg)


argImg = arg.reshape(img.shape[0:2])


import matplotlib,numpy
import pylab
# A random colormap for matplotlib
cmap = matplotlib.colors.ListedColormap ( numpy.random.rand ( argImg.max()+1,3))
pylab.imshow ( argImg.swapaxes(0,1), cmap = cmap)
pylab.show()

