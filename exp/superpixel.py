import vigra
import opengm
import numpy
import matplotlib.pyplot as plt
#img = vigra.readImage('/home/tbeier/datasets/BSR/BSDS500/data/images/train/56028.jpg')
img = vigra.readImage('/home/tbeier/datasets/BSR/BSDS500/data/images/val/145086.jpg')
imgLab = vigra.colors.transform_RGB2Lab(img)


show = 2

# get overseg
sp,nseg = vigra.analysis.slicSuperpixels(imgLab, 15.0, 20)
sp = vigra.analysis.labelImage(sp)-1
gridGraph = vigra.graphs.gridGraph(img.shape[0:2])
rag = vigra.graphs.regionAdjacencyGraph(gridGraph, sp)

print rag
if show<=1:
    rag.show(img)
    vigra.show()


# get edge indicator
imgLabBig = vigra.resize(imgLab, [imgLab.shape[0]*2-1, imgLab.shape[1]*2-1])
gradMag = vigra.filters.gaussianGradientMagnitude(imgLabBig,8.0).squeeze()


expImg =numpy.exp(gradMag**(0.5))**2

expImg2 =numpy.exp(-0.01*gradMag)


wImg = expImg

if show<=2:
    vigra.imshow(wImg)
    vigra.show()



gridGraphEdgeIndicator = vigra.graphs.edgeFeaturesFromInterpolatedImage(gridGraph,
                                                                  expImg)
gridGraphEdgeIndicator2 = vigra.graphs.edgeFeaturesFromInterpolatedImage(gridGraph,
                                                                  expImg2)


# accumulate edge indicator
weights = rag.accumulateEdgeFeatures(gridGraphEdgeIndicator)
weights2 = rag.accumulateEdgeFeatures(gridGraphEdgeIndicator2)
weights = weights
pathFinder = vigra.graphs.ShortestPathPathDijkstra(rag)

assert rag.nodeNum == rag.maxNodeId +1

dists = numpy.zeros([rag.nodeNum,rag.nodeNum], dtype=numpy.float64)
# run all pairs of shortest path
for i in range(rag.maxNodeId):
    #print i

    pathFinder.run(weights,rag.nodeFromId(i))
    distances = pathFinder.distances()
    distances[i]=0.0
    dists[i,:]=distances[:]
    resImg = rag.projectNodeFeaturesToGridGraph(distances)
    resImg = vigra.taggedView(resImg, "xy")

    #vigra.imshow(resImg)
    #vigra.show()


gm = opengm.graphicalModel(  numpy.ones([rag.nodeNum],dtype=opengm.label_type)*rag.nodeNum)


edgeDict = dict()

nVar = rag.nodeNum
for ei in rag.edgeIter():
    uv = sorted([rag.uId(ei),rag.vId(ei)])
    edgeDict[tuple(uv)] = True

    pf = opengm.pottsFunction([nVar,nVar],0.0, 100.0*float(weights2[ei.id]))
    gm.addFactor(gm.addFunction(pf),uv)

for vi0 in range(rag.maxNodeId):
    for vi1 in range(vi0+1,rag.maxNodeId):
        uv = (vi0, vi1)
        if uv in edgeDict :
            pass
        else:      
            pf = opengm.pottsFunction([nVar,nVar],0.0, -1.0*float(dists[vi0, vi1]))
            gm.addFactor(gm.addFunction(pf),uv)


verbose = True
useQpbo = False
useCgc = True
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
        randomizer = opengm.weightRandomizer(noiseType='normalAdd',noiseParam=50.700000001, ignoreSeed=False),
        stopWeight=0.0,
        reduction=0.95,
        setCutToZero=False
    )
    


    infParam = opengm.InfParam(
        numStopIt=5,
        numIt=200,
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
            doCutMove = True,
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


floatArg = arg.astype(numpy.float32)
rImg = rag.projectNodeFeaturesToGridGraph(floatArg)
rImg = vigra.taggedView(rImg, "xy")

rag.show(img, arg.astype(numpy.uint32)+1)
vigra.show()
vigra.imshow((rImg))
vigra.show()
