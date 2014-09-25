import vigra
import opengm
import numpy
import matplotlib
import matplotlib.pyplot as plt
img = vigra.readImage('/home/tbeier/datasets/BSR/BSDS500/data/images/train/56028.jpg')
img = vigra.readImage('/home/tbeier/datasets/BSR/BSDS500/data/images/train/118035.jpg')
img = vigra.resize(img, (img.shape[0]/3, img.shape[1]/3))[::2,::2,:]



grad = vigra.filters.gaussianGradientMagnitude(vigra.colors.transform_Lab2RGB(img), 1.5).squeeze()



attractive  = numpy.exp(-0.0005*grad)+0.5
repulsive  = (1.0 - attractive)*10.0


imgplot = plt.imshow(attractive.swapaxes(0,1))
plt.colorbar()
plt.show()

imgplot = plt.imshow(1.0*repulsive.swapaxes(0,1))
plt.colorbar()
plt.show()





gm = opengm.adder.gridPatchAffinityGm(
    repulsive.astype(numpy.float64), 
    attractive.astype(numpy.float64), 
    12, 3 ,3, 0.0, 10000.40
)
print gm

verbose = True
useQpbo = True
useCgc = False
useWs = True

#with opengm.Timer("with cgc"):
#    infParam = opengm.InfParam(
#        planar=False,
#        startFromThreshold=False,
#        doCutMove = True,
#        doGlueCutMove = True,
#        maxIterations = 1
#    )
#    #inf=opengm.inference.Cgc(gm, parameter=infParam)
#    inf=opengm.inference.Multicut(gm)
#    inf.infer(inf.verboseVisitor())
#
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
        randomizer = opengm.weightRandomizer(noiseType='normalAdd',noiseParam=10.700000001, ignoreSeed=False),
        stopWeight=0.0,
        reduction=0.90,
        setCutToZero=False
    )
    


    infParam = opengm.InfParam(
        numStopIt=2,
        numIt=10,
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
            seedFraction = 0.010
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

