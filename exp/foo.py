import vigra
import opengm
import numpy
import matplotlib
import matplotlib.pyplot as plt

img = vigra.readImage('/home/tbeier/datasets/BSR/BSDS500/data/images/train/56028.jpg')
img = vigra.readImage('/home/tbeier/datasets/BSR/BSDS500/data/images/train/118035.jpg')
#img = vigra.resize(img, (img.shape[0]/1, img.shape[1]/1))#[::1,::1,:]



grad = vigra.filters.gaussianGradientMagnitude(vigra.colors.transform_Lab2RGB(img), 1.5).squeeze()



attractive  = numpy.exp(-0.0005*grad)
repulsive  = (1.0 - attractive)*4.5


imgplot = plt.imshow(attractive.swapaxes(0,1))
plt.colorbar()
plt.show()

imgplot = plt.imshow(1.0*repulsive.swapaxes(0,1))
plt.colorbar()
plt.show()





gm = opengm.adder.gridPatchAffinityGm(
    repulsive.astype(numpy.float64), 
    attractive.astype(numpy.float64), 
    20, 5 ,5, 1.5, 2.2
)
print gm

verbose = True
useQpbo = False
useRC = False
useCgc = True
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

    fusionParam = opengm.InfParam(fusionSolver = 'multicut', planar=False)
    arg = None


    if useWs:
        print "ws"
        proposalParam = opengm.InfParam(
            randomizer = opengm.weightRandomizer(noiseType='normalAdd',noiseParam=10.100000001,ignoreSeed=False),
            ignoreNegativeWeights = True,
            seedFraction = 100.0
        )
        infParam = opengm.InfParam(
            numStopIt=400,
            numIt=2000,
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

    if useRC:
        proposalParam = opengm.InfParam(
            randomizer = opengm.weightRandomizer(noiseType='normalAdd',noiseParam=1.700000001, ignoreSeed=False),
            stopWeight=-100.0,
            nodeStopNum=200,
            ignoreNegativeWeights=True,
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

