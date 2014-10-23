import opengm
import numpy
#---------------------------------------------------------------
# MinSum  with SelfFusion
#---------------------------------------------------------------
numpy.random.seed(42)

#gm=opengm.loadGm("/home/tbeier/datasets/image-seg/3096.bmp.h5","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/image-seg/175032.bmp.h5","gm")
gm=opengm.loadGm("/media/tbeier/TOSHIBA EXT/bigger_mmwc_gm","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/image-seg/148026.bmp.h5","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/knott-3d-450/gm_knott_3d_102.h5","gm")#(ROTTEN)
#gm=opengm.loadGm("/home/tbeier/datasets/knott-3d-450/gm_knott_3d_096.h5","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/knott-3d-300/gm_knott_3d_078.h5","gm")
#gm=opengm.loadGm("/home/tbeier/datasets/knott-3d-150/gm_knott_3d_038.h5","gm")

#---------------------------------------------------------------
# Minimize
#---------------------------------------------------------------
#get an instance of the optimizer / inference-algorithm



print gm




verbose = True
useRC = True


with opengm.Timer("with new method"):

    fusionParam = opengm.InfParam(fusionSolver = 'multicut', planar=False)
    randomizer = opengm.weightRandomizer(noiseType='normalAdd',noiseParam=1.500000001,ignoreSeed=False)
    parallelProposals = 1
    arg = None




    if useRC:
        proposalParam = opengm.InfParam(
            randomizer = randomizer,
            stopWeight=-1000000.0,
            nodeStopNum=0.01,
            ignoreNegativeWeights=False,
            setCutToZero=False
        )
        
        bv =  opengm.BoolVector()
        bv.append(True)
        bv.append(True)
        bv.append(False)
        infParam = opengm.InfParam(
            numStopIt=50,
            numIt=50,
            generator='randomizedHierarchicalClustering',
            proposalParam=proposalParam,
            fusionParam = fusionParam,
            parallelProposals=parallelProposals,
            allowCutsWithin = bv
        )


        inf=opengm.inference.IntersectionBased(gm, parameter=infParam)
        visitor=inf.verboseVisitor(printNth=1,multiline=False)
        if verbose:
            inf.infer(visitor)
        else:
            inf.infer()
        arg = inf.arg()












