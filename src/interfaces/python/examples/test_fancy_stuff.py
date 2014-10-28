import opengm
import numpy
#---------------------------------------------------------------
# MinSum  with SelfFusion
#---------------------------------------------------------------
numpy.random.seed(42)

#gm=opengm.loadGm("/home/tbeier/datasets/image-seg/3096.bmp.h5","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/image-seg/175032.bmp.h5","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/image-seg/291000.bmp.h5","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/image-seg/148026.bmp.h5","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/knott-3d-450/gm_knott_3d_102.h5","gm")#(ROTTEN)
gm=opengm.loadGm("/home/tbeier/datasets/knott-3d-450/gm_knott_3d_096.h5","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/knott-3d-300/gm_knott_3d_078.h5","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/knott-3d-150/gm_knott_3d_038.h5","gm")

#---------------------------------------------------------------
# Minimize
#---------------------------------------------------------------
#get an instance of the optimizer / inference-algorithm



print gm

verbose = True

with opengm.Timer("with new method"):

    fusionParam = opengm.InfParam(fusionSolver = 'multicut', planar=False, decompose=True)
    randomizer = opengm.weightRandomizer(noiseType='normalAdd',noiseParam=1.500000001,ignoreSeed=False)
    parallelProposals = 8
    arg = None




    proposalParam = opengm.InfParam(
    randomizer = randomizer,
    stopWeight=-0.0,
    nodeStopNum=0.01,
    ignoreNegativeWeights=False,
    setCutToZero=False
    )



    infParam = opengm.InfParam(
    numStopIt=10,
    numIt=10,
    generator='randomizedHierarchicalClustering',
    proposalParam=proposalParam,
    fusionParam = fusionParam,
    parallelProposals=parallelProposals
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

   
print gm.evaluate(arg)

