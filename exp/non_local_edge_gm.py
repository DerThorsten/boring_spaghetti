import opengm
import vigra




class PixelMulticut(object):
    def __init__(self, shape):
        self.shape = shape


    def addLocalWeights(self, cutCost, cutBenefit=None):
        """
            cutCost:
                these cutCost should ONLY encourage merging.
                The weight for NOT merging can be 0 if there is
                *STRONG* indicator for boundary.
                Therefore cutCost should be >=0.
                (can be negative, but one should use cut Benefit
                 for that)
                In practice cutCost can be
                a strict positive image as gaussianGradientMagnitude
                filter applied to the image

            cutBenefit:
                a sparse image which is zero in every pixel
                where we do not want a label transition.
                if the pixel is negative, a label transition
                will be encouraged.
                Furthermore, if the pixel is negative, all
                values of cutCost are overwritten.
        """
        pass

    def addNonLocalCannotLinkConstraints(self, constraints):
        """ add "hard" constraints 
        (implemented as soft constraints with high energy)
        to NOT link a pair of pixels.
            constraints:
                a |C| x 4 numpy array
                xa1,ya1,xb1:yb1,
                xa2,ya2,xb2:yb2
                ...
                xa|C|,ya|C|,xb|C|:yb|C|
        """
        pass

    def addSlicLikeSeeds(self,seedDist):
        pass

    def selfAddMaxSizeWeights(self, size, weights):
        pass


    def addPatchWeights(self, image, edgeImage, maxRad=10, minRad=1, stepRad=2):
        """ add semi local weights within a path which ONLY encourage label transitions
            image :
                RGB or feature image to compute "color" difference on
            edgeImage :
                image to compute max intervening edge weight on.
                Via line or shortest path? Lets make a weighted combination

            maxRad :
                max radius 

            minRad :
                min radius

            stepRad: 
                radius increment (maybe non linear is better)

        """
        pass


    def addGlobalWeights(self, image, edgeImage):
        """ add sparse global edges which ONLY encourage label transitions
        """
        pass
