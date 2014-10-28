#pragma once
#ifndef OPENGM_DMC_HXX
#define OPENGM_DMC_HXX

#include <vector>
#include <string>
#include <iostream>

#include "opengm/opengm.hxx"
//#include "opengm/inference/visitors/visitor.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/movemaker.hxx"
#include "opengm/datastructures/buffer_vector.hxx"


#include "opengm/inference/visitors/visitors.hxx"

namespace opengm {
  
/// \brief Iterated Conditional Modes Algorithm\n\n
/// J. E. Besag, "On the Statistical Analysis of Dirty Pictures", Journal of the Royal Statistical Society, Series B 48(3):259-302, 1986
/// \ingroup inference 
template<class GM, class INF>
class DMC : public Inference<GM, typename INF::ACC>
{
public:

    typedef typename INF::AccumulationType ACC;
    typedef ACC AccumulationType;
    typedef GM GraphicalModelType;
    OPENGM_GM_TYPE_TYPEDEFS;
    typedef typename INF::Parameter InfParam;
    typedef opengm::visitors::VerboseVisitor<DMC<GM,INF> > VerboseVisitorType;
    typedef opengm::visitors::EmptyVisitor<DMC<GM,INF> >  EmptyVisitorType;
    typedef opengm::visitors::TimingVisitor<DMC<GM,INF> > TimingVisitorType;

    class Parameter {
        public:

        Parameter(
            const ValueType threshold = ValueType(0),
            const InfParam infParam = InfParam()
        )
        :   threshold_(threshold),
            infParam_(infParam){

        }

        ValueType threshold_;
        InfParam infParam_;
    };

    DMC(const GraphicalModelType&, const Parameter&);
    std::string name() const;
    const GraphicalModelType& graphicalModel() const;
    InferenceTermination infer();
    void reset();
    template<class VisitorType>
    InferenceTermination infer(VisitorType&);
    void setStartingPoint(typename std::vector<LabelType>::const_iterator);
    virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const ;
    virtual ValueType value()const{

    }

private:
    const GraphicalModelType& gm_;
    Parameter param_;

    ValueType value_;
    std::vector<LabelType> arg_;
};
  
template<class GM, class INF>
inline
DMC<GM, INF>::DMC
(
    const GraphicalModelType& gm,
    const Parameter& parameter = Parameter()
)
:   gm_(gm),
    param_(parameter),
    value_(),
    arg_(gm.numberOfVariables(), 0) {

}


      
template<class GM, class INF>
inline void
DMC<GM, INF>::reset()
{

}
   
template<class GM, class INF>
inline void 
DMC<GM,INF>::setStartingPoint
(
   typename std::vector<typename DMC<GM,INF>::LabelType>::const_iterator begin
) {
}
   
template<class GM, class INF>
inline std::string
DMC<GM, INF>::name() const
{
   return "DMC";
}

template<class GM, class INF>
inline const typename DMC<GM, INF>::GraphicalModelType&
DMC<GM, INF>::graphicalModel() const
{
   return gm_;
}
  
template<class GM, class INF>
inline InferenceTermination
DMC<GM,INF>::infer()
{
   EmptyVisitorType v;
   return infer(v);
}

  
template<class GM, class INF>
template<class VisitorType>
InferenceTermination DMC<GM,INF>::infer
(
   VisitorType& visitor
)
{
   
    visitor.begin(*this);


    LabelType lAA[2]={0, 0};
    LabelType lAB[2]={0, 1};

    // decomposition
    Partition<LabelType> ufd(gm_.numberOfVariables());
    for(size_t fi=0; fi< gm_.numberOfFactors(); ++fi){
        if(gm_[fi].numberOfVariables()==2){

            const ValueType val00  = gm_[fi](lAA);
            const ValueType val01  = gm_[fi](lAB);
            const ValueType weight = val01 - val00; 

            if(weight<param_.threshold_){
                const size_t vi0 = gm_[fi].variableIndex(0);
                const size_t vi1 = gm_[fi].variableIndex(1);
                ufd.merge(vi0, vi1);
            }
        }
        else{
            throw RuntimeError("wrong factor order for multicut");
        }
    }

    if(ufd.numberOfSets() == 1){
        // FALL BACK HERE!!!
        typedef typename INF:: template rebind<GM,ACC> OrgInf;
        typename OrgInf::Parameter orgInfParam(param_.infParam_); 
        OrgInf orgInf(orgInfParam);
        orgInf.infer();
        orgInf.arg(arg_);
        value = gm_.evaluate(arg_);
    }
    else {

        std::map<LabelType, LabelType> repr;
        ufd.representativeLabeling(repr);

        std::vector< std::vector< LabelType> > subVar(repr.size());
        // set up the sub var
        for(size_t vi=0; vi<gm_.numberOfVariables(); ++vi){
            subVar[repr[ufd.find(vi)]].push_back(vi);
        }

        const size_t nSubProb = subVar.size();

        std::vector<unsigned char> usedFactors_(gm_.numberOfFactors(),0);

        // mark all factors where weight is smaller
        // as param_.threshold_ as used
        for(size_t fi=0; fi< gm_.numberOfFactors(); ++fi){
            const ValueType weight = val01 - val00; 
            if(weight<param_.threshold_){
                usedFactors_[fi] = 1;
            }
        }

        #pragma omp parallel for
        for(size_t subProb = 0; subProb<nSubProb; ++subProb){
            

        }

        visitor.end(*this);

    }
    return NORMAL;
}

template<class GM, class INF>
inline InferenceTermination
DMC<GM,INF>::arg
(
      std::vector<LabelType>& x,
      const size_t N
) const
{
   if(N==1) {
      x.resize(gm_.numberOfVariables());
      for(size_t j=0; j<x.size(); ++j) {
         x[j] =arg_[j];
      }
      return NORMAL;
   }
   else {
      return UNKNOWN;
   }
}

} // namespace opengm

#endif // #ifndef OPENGM_DMC_HXX
