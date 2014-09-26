#ifndef INTERSECTION_BASED_PARAM
#define INTERSECTION_BASED_PARAM

#include "empty_param.hxx"
#include "param_exporter_base.hxx"
//solver specific
#include "opengm/inference/intersection_based_inf.hxx"

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterIntersectionBased{

public:
    typedef typename INFERENCE::ValueType ValueType;
    typedef typename INFERENCE::Parameter Parameter;
    typedef typename INFERENCE::ProposalGen Gen;
    typedef typename Gen::Parameter GenParameter;
    typedef typename INFERENCE::FusionParameter FusionParameter;
    typedef InfParamExporterIntersectionBased<INFERENCE> SelfType;

    inline static void set 
    (
        Parameter & p,
        const GenParameter & proposalParam,
        const FusionParameter & fusionParam,
        const size_t         numIt,
        const size_t         numStopIt
    ) {
        p.proposalParam_ = proposalParam;
        p.fusionParam_ = fusionParam;
        p.numIt_ = numIt;
        p.numStopIt_ = numStopIt;
    } 

    void static exportInfParam(const std::string & className){


    class_<Parameter > ( className.c_str() , init< > ())
        .def_readwrite("proposalParam",&Parameter::proposalParam_,"parameters of the proposal generator")
        .def_readwrite("fusionParam",&Parameter::fusionParam_,"parameters of the fusion move solver")
        .def_readwrite("numIt",&Parameter::numIt_,"total number of iterations")
        .def_readwrite("numStopIt",&Parameter::numStopIt_,"stop after n not successful steps")
        .def ("set", &SelfType::set, 
            (
                boost::python::arg("proposalParam")=GenParameter(),
                boost::python::arg("fusionParam")=FusionParameter(),
                boost::python::arg("numIt")=1000, 
                boost::python::arg("numStopIt")=0
            ) 
        )
    ;


    }
};


template<class GM, class ACC>
class InfParamExporter<
    opengm::PermutableLabelFusionMove<GM, ACC>
>{

public:
    typedef opengm::PermutableLabelFusionMove<GM, ACC> FM;
    typedef typename FM::ValueType ValueType;
    typedef typename FM::Parameter Parameter;
    typedef InfParamExporter< FM > SelfType;   




    inline static void set 
    (
        Parameter & p,
        const typename FM::FusionSolver fusionSolver,
        const bool planar
    ) {
        p.fusionSolver_ = fusionSolver;
        p.planar_ = planar;

    } 

    void static exportInfParam(const std::string & className){




        enum_<typename FM::FusionSolver> ("_IntersectionBased_FusionMover_FusionSolverEnum_")
          .value("default", FM::DefaultSolver)
          .value("multicut", FM::MulticutSolver)
          .value("cgc", FM::CgcSolver)
          .value("hc", FM::HierachicalClusteringSolver)
       ;



        class_<Parameter > ( className.c_str() , init< > ())
            .def_readwrite("fusionSolver",&Parameter::fusionSolver_,"fusionSolver parameter")
            .def_readwrite("planar",&Parameter::planar_,"planar")
            .def ("set", &SelfType::set, 
                (
                    boost::python::arg("fusionSolver")= typename FM::FusionSolver(),
                    boost::python::arg("planar")=false
                ) 
            )
        ;
    }
};









#ifndef NOVIGRA
template<class GM, class ACC>
class InfParamExporter<
    opengm::proposal_gen::RandomizedHierarchicalClustering<GM, ACC>
>{

public:
    typedef opengm::proposal_gen::RandomizedHierarchicalClustering<GM, ACC> GEN;
    typedef typename GEN::ValueType ValueType;
    typedef typename GEN::Parameter Parameter;
    typedef InfParamExporter< GEN > SelfType;   

    typedef typename opengm:: proposal_gen::WeightRandomization<ValueType>::Parameter WRandParam;


    inline static void set 
    (
        Parameter & p,
        const WRandParam &             randomizer,
        const float                    stopWeight,
        const float                    reduction,
        const bool                     setCutToZero
    ) {
        p.randomizer_ = randomizer;
        p.stopWeight_ = stopWeight;
        p.reduction_ = reduction;
        p.setCutToZero_ = setCutToZero;
    } 

    void static exportInfParam(const std::string & className){



    class_<Parameter > ( className.c_str() , init< > ())
        .def_readwrite("randomizer",&Parameter::randomizer_,"weight randomizer parameter")
        .def_readwrite("stopWeight",&Parameter::stopWeight_,"stopWeight")
        .def_readwrite("reduction",&Parameter::reduction_,"reduction")
        .def_readwrite("setCutToZero",&Parameter::setCutToZero_,"set weights of cut edge to zero")
        .def ("set", &SelfType::set, 
            (
                boost::python::arg("randomizer")=WRandParam(),
                boost::python::arg("stopWeight")=0.0,
                boost::python::arg("reduction")=-1.0,
                boost::python::arg("setCutToZero")=true
            ) 
        )
    ;
    }
};

template<class GM, class ACC>
class InfParamExporter<
    opengm::proposal_gen::RandomizedWatershed<GM, ACC>
>{

public:
    typedef opengm::proposal_gen::RandomizedWatershed<GM, ACC> GEN;
    typedef typename GEN::ValueType ValueType;
    typedef typename GEN::Parameter Parameter;
    typedef InfParamExporter< GEN > SelfType;

    typedef typename opengm:: proposal_gen::WeightRandomization<ValueType>::Parameter WRandParam;

    inline static void set 
    (
        Parameter & p,
        const float                    seedFraction,
        const bool                     ignoreNegativeWeights,
        const WRandParam &             randomizer
    ) {
        p.seedFraction_ = seedFraction;
        p.ignoreNegativeWeights_ = ignoreNegativeWeights;
        p.randomizer_ = randomizer;
    } 

    void static exportInfParam(const std::string & className){




    class_<Parameter > ( className.c_str() , init< > ())
        .def_readwrite("seedFraction",&Parameter::seedFraction_,"approximative relative size of seeds (or absolute seed number if >1")
        .def_readwrite("ignoreNegativeWeights",&Parameter::ignoreNegativeWeights_,"ignoreNegativeWeights")
        .def_readwrite("randomizer",&Parameter::randomizer_,"weight randomizer")
        .def ("set", &SelfType::set, 
            (
                boost::python::arg("seedFraction")=0.01,
                boost::python::arg("ignoreNegativeWeights")=false,
                boost::python::arg("randomizer")=WRandParam()
            ) 
        )
    ;
    }
};

#endif





template<class INFERENCE>
class InfParamExporterEmpty;


#define _EMPTY_PROPOSAL_PARAM(clsName)                        \
template<class GM,class ACC>                                  \
class InfParamExporter<          clsName <GM,ACC>     >       \
: public  InfParamExporterEmpty< clsName < GM,ACC>    > {     \
};
#ifdef WITH_QPBO
_EMPTY_PROPOSAL_PARAM(opengm::proposal_gen::QpboBased)
#endif
//_EMPTY_PROPOSAL_PARAM(opengm::proposal_gen::Random2Gen)
#undef _EMPTY_PROPOSAL_PARAM



template<class GM,class ACC>
class InfParamExporter<opengm::IntersectionBasedInf<GM,ACC> >  : public  InfParamExporterIntersectionBased<opengm::IntersectionBasedInf< GM,ACC> > {

};

#endif
