#ifndef INTERSECTION_BASED_CALLER
#define INTERSECTION_BASED_CALLER

#include <opengm/opengm.hxx>
#include <opengm/inference/intersection_based_inf.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class IntersectionBasedCaller : public InferenceCallerBase<IO, GM, ACC, IntersectionBasedCaller<IO, GM, ACC> > {
protected:

   typedef InferenceCallerBase<IO, GM, ACC, IntersectionBasedCaller<IO, GM, ACC> > BaseClass;


   typedef typename BaseClass::OutputBase OutputBase;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer; 

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   template<class INF>
   void setParam(typename INF::Parameter & param);

   size_t numIt_;
   size_t numStopIt_;
   size_t parallelProposals_;

   int numberOfThreads_;
   std::string selectedGenType_;
   std::string selectedFusionType_;


   // RHC SPECIFIC param

   
public:
   const static std::string name_;
   IntersectionBasedCaller(IO& ioIn);
   ~IntersectionBasedCaller();
};

template <class IO, class GM, class ACC>
inline  IntersectionBasedCaller<IO, GM, ACC>::IntersectionBasedCaller(IO& ioIn)
: BaseClass(name_, "detailed description of  SelfFusion caller...", ioIn) { 
   std::vector<std::string> fusion;
   fusion.push_back("MC");
   fusion.push_back("CGC");
   fusion.push_back("HC");
   std::vector<std::string> gen;  
   gen.push_back("RHC");
   gen.push_back("R2C");
   gen.push_back("RWS");


   addArgument(StringArgument<>(selectedGenType_, "g", "gen", "Selected proposal generator", gen.front(), gen));
   addArgument(StringArgument<>(selectedFusionType_, "f", "fusion", "Select fusion method", fusion.front(), fusion));
   //addArgument(IntArgument<>(numberOfThreads_, "", "threads", "number of threads", static_cast<int>(1)));
   addArgument(Size_TArgument<>(numIt_, "", "numIt", "number of iterations", static_cast<size_t>(100))); 
   addArgument(Size_TArgument<>(numStopIt_, "", "numStopIt", "number of iterations with no improvment that cause stopping (0=auto)", static_cast<size_t>(0))); 
   addArgument(Size_TArgument<>(parallelProposals_, "pp", "parallelProposals", "number of parallel proposals (1)", static_cast<size_t>(1))); 


}

template <class IO, class GM, class ACC>
IntersectionBasedCaller<IO, GM, ACC>::~IntersectionBasedCaller()
{;}


template <class IO, class GM, class ACC>
template <class INF>
inline void IntersectionBasedCaller<IO, GM, ACC>::setParam(
   typename INF::Parameter & param
){

   param.numIt_ = numIt_;
   param.numStopIt_ = numStopIt_;  
}

template <class IO, class GM, class ACC>
inline void IntersectionBasedCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running Intersection Based caller" << std::endl;


   
   typedef opengm::proposal_gen::RandomizedHierarchicalClustering<GM, opengm::Minimizer> RHC;
   typedef opengm::proposal_gen::RandomizedWatershed<GM, opengm::Minimizer> RWS;
   typedef opengm::proposal_gen::QpboBased<GM, opengm::Minimizer> R2C;




   if(selectedGenType_=="RHC"){
      typedef RHC Gen;
      typedef opengm::IntersectionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      //para.proposalParam_.sigma_ = sigma_;
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }
   else if(selectedGenType_=="RWS"){
      typedef RWS Gen;
      typedef opengm::IntersectionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      //para.proposalParam_.sigma_ = sigma_;
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }
   else if(selectedGenType_=="R2C"){
      typedef R2C Gen;
      typedef opengm::IntersectionBasedInf<GM, Gen> INF;
      typename INF::Parameter para;
      setParam<INF>(para);
      //para.proposalParam_.sigma_ = sigma_;
      this-> template infer<INF, typename INF::TimingVisitorType, typename INF::Parameter>(model, output, verbose, para);
   }
}

template <class IO, class GM, class ACC>
const std::string  IntersectionBasedCaller<IO, GM, ACC>::name_ = "IntersectionBased";

} // namespace interface

} // namespace opengm

#endif /* INTERSECTION_BASED_CALLER */
