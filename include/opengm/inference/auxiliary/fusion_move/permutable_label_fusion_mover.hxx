#ifndef OPENGM_PERMUTABLE_LABEL_FUSION_MOVER_HXX
#define OPENGM_PERMUTABLE_LABEL_FUSION_MOVER_HXX


#include <opengm/inference/inference.hxx>
#include <opengm/inference/multicut.hxx>

// FIXME
using namespace std;
#define Isinf Isinf2
#include <opengm/inference/cgc.hxx>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>


#ifndef NOVIGRA
#include <vigra/adjacency_list_graph.hxx>
#include <vigra/merge_graph_adaptor.hxx>
#include <vigra/hierarchical_clustering.hxx>
#include <vigra/priority_queue.hxx>
#include <vigra/random.hxx>
#include <vigra/graph_algorithms.hxx>

#endif

namespace opengm{







    #ifndef NOVIGRA
    template<class GM, class ACC >
    class McClusterOp{
    public:
        typedef ACC AccumulationType;
        typedef GM GraphicalModelType;
        OPENGM_GM_TYPE_TYPEDEFS;

        typedef vigra::AdjacencyListGraph Graph;
        typedef vigra::MergeGraphAdaptor< Graph > MergeGraph;


        typedef typename MergeGraph::Edge Edge;
        typedef ValueType WeightType;
        typedef IndexType index_type;
        struct Parameter
        {



            Parameter(
                const float stopWeight = 0.0
            )
            :
                stopWeight_(stopWeight){
            }
            float stopWeight_;
        };


        McClusterOp(const Graph & graph , 
                    MergeGraph & mergegraph, 
                    const Parameter & param,
                    std::vector<ValueType> & weights
                   )
        :
            graph_(graph),
            mergeGraph_(mergegraph),
            pq_(graph.edgeNum()),
            param_(param),
            weights_(weights){

            for(size_t i=0; i<graph_.edgeNum(); ++i){
                size_t u = graph_.id(graph_.u(graph_.edgeFromId(i)));
                size_t v = graph_.id(graph_.v(graph_.edgeFromId(i)));
                pq_.push(i, weights_[i]);
            }
        }




        void reset(){
            pq_.reset();
        }



        Edge contractionEdge(){
            index_type minLabel = pq_.top();
            while(mergeGraph_.hasEdgeId(minLabel)==false){
                pq_.deleteItem(minLabel);
                minLabel = pq_.top();
            }
            return Edge(minLabel);
        }

        /// \brief get the edge weight of the edge which should be contracted next
        WeightType contractionWeight(){
            index_type minLabel = pq_.top();
            while(mergeGraph_.hasEdgeId(minLabel)==false){
                pq_.deleteItem(minLabel);
                minLabel = pq_.top();
            }
            return pq_.topPriority();

        }

        /// \brief get a reference to the merge
        MergeGraph & mergeGraph(){
            return mergeGraph_;
        }

        bool done()const{
            return pq_.topPriority()<=ValueType(param_.stopWeight_);
        }

        void mergeEdges(const Edge & a,const Edge & b){
            weights_[a.id()]+=weights_[b.id()];
            pq_.push(a.id(), weights_[a.id()]);
            pq_.deleteItem(b.id());
        }

        void eraseEdge(const Edge & edge){
            pq_.deleteItem(edge.id());
        }

        const Graph & graph_;
        MergeGraph & mergeGraph_;
        vigra::ChangeablePriorityQueue< ValueType ,std::greater<ValueType> > pq_;
        Parameter param_;
        std::vector<ValueType> & weights_;
    };


    #endif





template<class GM, class ACC>
class PermutableLabelFusionMove{

public:

    typedef GM GraphicalModelType;
    typedef ACC AccumulationType;

    typedef std::map<UInt64Type, float> MapType;
    typedef typename MapType::iterator MapIter;
    typedef typename MapType::const_iterator MapCIter;

    OPENGM_GM_TYPE_TYPEDEFS;

    typedef PermutableLabelFusionMove<GM, ACC> SelfType;

    enum FusionSolver{
        DefaultSolver,
        MulticutSolver,
        CgcSolver,
        HierachicalClusteringSolver
    };

    struct Parameter{
        Parameter(
            const FusionSolver fusionSolver = SelfType::DefaultSolver,
            const bool planar = false
        )
        : 
            fusionSolver_(fusionSolver),
            planar_(planar)
        {

        }
        FusionSolver fusionSolver_;
        bool planar_;

    };

    typedef SimpleDiscreteSpace<IndexType, LabelType>       SubSpace;
    typedef PottsFunction<ValueType, IndexType, LabelType>  PFunction;
    typedef GraphicalModel<ValueType, Adder, PFunction , SubSpace> SubModel;


    PermutableLabelFusionMove(const GraphicalModelType & gm, const Parameter & param = Parameter())
    :   
        gm_(gm),
        param_(param)
    {
        if(param_.fusionSolver_ == DefaultSolver){

            #ifdef WITH_CPLEX
                param_.fusionSolver_ = MulticutSolver;
            #endif
            
            if(param_.fusionSolver_ == DefaultSolver){
                #ifdef WITH_QPBO 
                    param_.fusionSolver_ = CgcSolver;
                #endif
            }
            if(param_.fusionSolver_ == DefaultSolver){
                #ifdef WITH_ISINF
                    if(param_.planar_){
                        param_.fusionSolver_ = CgcSolver;
                    }
                #endif
            }
            if(param_.fusionSolver_ == DefaultSolver){
                #ifndef NOVIGRA
                    if(param_.planar_){
                        param_.fusionSolver_ = HierachicalClusteringSolver;
                    }
                #endif
            }
            if(param_.fusionSolver_ == DefaultSolver){
                throw RuntimeError("WITH_CPLEX or WITH_QPBO or WITH_ISINF must be enabled");
            }
        }
    }



    void printArg(const std::vector<LabelType> arg) {
         const size_t nx = 3; // width of the grid
        const size_t ny = 3; // height of the grid
        const size_t numberOfLabels = nx*ny;

        size_t i=0;
        for(size_t y = 0; y < ny; ++y){
            
            for(size_t x = 0; x < nx; ++x) {
                std::cout<<arg[i]<<"  ";
            }
            std::cout<<"\n";
            ++i;
        }
        
    }


    size_t intersect(
        const std::vector<LabelType> & a,
        const std::vector<LabelType> & b,
        std::vector<LabelType> & res
    ){
        Partition<LabelType> ufd(gm_.numberOfVariables());
        for(size_t fi=0; fi< gm_.numberOfFactors(); ++fi){
            if(gm_[fi].numberOfVariables()==2){

                const size_t vi0 = gm_[fi].variableIndex(0);
                const size_t vi1 = gm_[fi].variableIndex(1);



                if(a[vi0] == a[vi1] && b[vi0] == b[vi1]){
                    ufd.merge(vi0, vi1);
                }
            }   
            else{
                throw RuntimeError("only implemented for second order");
            }
        }
        std::map<LabelType, LabelType> repr;
        ufd.representativeLabeling(repr);
        for(size_t vi=0; vi<gm_.numberOfVariables(); ++vi){
            res[vi]=repr[ufd.find(vi)];
        }
        //std::cout<<" A\n";
        //printArg(a);
        //std::cout<<" B\n";
        //printArg(b);
        //std::cout<<" RES\n";
        //printArg(res);

        return ufd.numberOfSets();
    }

    bool fuse(
        const std::vector<LabelType> & a,
        const std::vector<LabelType> & b,
        std::vector<LabelType> & res,
        const ValueType valA,
        const ValueType valB,
        ValueType & valRes
    ){

        std::vector<LabelType> ab(gm_.numberOfVariables());
        IndexType numNewVar = this->intersect(a, b, ab);
        //std::cout<<"numNewVar "<<numNewVar<<"\n";

        if(numNewVar==1){
            return false;
        }

        const ValueType intersectedVal = gm_.evaluate(ab);



        // get the new smaller graph


        MapType accWeights;
        size_t erasedEdges = 0;
        size_t notErasedEdges = 0;


        LabelType lAA[2]={0, 0};
        LabelType lAB[2]={0, 1};




        for(size_t fi=0; fi< gm_.numberOfFactors(); ++fi){
            if(gm_[fi].numberOfVariables()==2){
                const size_t vi0 = gm_[fi].variableIndex(0);
                const size_t vi1 = gm_[fi].variableIndex(1);

                const size_t cVi0 = ab[vi0] < ab[vi1] ? ab[vi0] : ab[vi1];
                const size_t cVi1 = ab[vi0] < ab[vi1] ? ab[vi1] : ab[vi0];

                OPENGM_CHECK_OP(cVi0,<,gm_.numberOfVariables(),"");
                OPENGM_CHECK_OP(cVi1,<,gm_.numberOfVariables(),"");


                if(cVi0 == cVi1){
                    ++erasedEdges;
                }
                else{
                    ++notErasedEdges;

                    // get the weight
                    ValueType val00  = gm_[fi](lAA);
                    ValueType val01  = gm_[fi](lAB);
                    ValueType weight = val01 - val00; 

                    //std::cout<<"vAA"<<val00<<" vAB "<<val01<<" w "<<weight<<"\n";

                    // compute key
                    const UInt64Type key = cVi0 + numNewVar*cVi1;

                    // check if key is in map
                    MapIter iter = accWeights.find(key);

                    // key not yet in map
                    if(iter == accWeights.end()){
                        accWeights[key] = weight;
                    }
                    // key is in map 
                    else{
                        iter->second += weight;
                    }

                }

            }
        }
        OPENGM_CHECK_OP(erasedEdges+notErasedEdges, == , gm_.numberOfFactors(),"something wrong");
        //std::cout<<"erased edges      "<<erasedEdges<<"\n";
        //std::cout<<"not erased edges  "<<notErasedEdges<<"\n";
        //std::cout<<"LEFT OVER FACTORS "<<accWeights.size()<<"\n";



        if(param_.fusionSolver_ == CgcSolver){
            return doMoveCgc(accWeights,ab,numNewVar, a, b, res, valA, valB, valRes);
        }
        else if(param_.fusionSolver_ == MulticutSolver){
            return doMoveMulticut(accWeights,ab,numNewVar, a, b, res, valA, valB, valRes);
        }
        else if(param_.fusionSolver_ == HierachicalClusteringSolver){
            return doMoveHierachicalClustering(accWeights,ab,numNewVar, a, b, res, valA, valB, valRes);
        }
        else{
            throw RuntimeError("unknown fusionSolver");
            return false;
        }
           
    }



    bool doMoveCgc(
        const MapType & accWeights,
        const std::vector<LabelType> & ab,
        const IndexType numNewVar,
        const std::vector<LabelType> & a,
        const std::vector<LabelType> & b,
        std::vector<LabelType> & res,
        const ValueType valA,
        const ValueType valB,
        ValueType & valRes
    ){


        // make the actual sub graphical model
        SubSpace subSpace(numNewVar, numNewVar);
        SubModel subGm(subSpace);

        // reserve space
        subGm. template reserveFunctions<PFunction>(accWeights.size());
        subGm.reserveFactors(accWeights.size());
        subGm.reserveFactorsVarialbeIndices(accWeights.size()*2);

        for(MapCIter i = accWeights.begin(); i!=accWeights.end(); ++i){
            const UInt64Type key    = i->first;
            const ValueType weight = i->second;

            const UInt64Type cVi1 = key/numNewVar;
            const UInt64Type cVi0 = key - cVi1*numNewVar;
            const UInt64Type vis[2] = {cVi0, cVi1};

            PFunction pf(numNewVar, numNewVar, 0.0, weight);
            subGm.addFactor(subGm.addFunction(pf), vis, vis+2);
        }

        std::vector<LabelType> subArg;

        //::cout<<"WITH MC\n";
        typedef CGC<SubModel, Minimizer> Inf;
        typedef  typename  Inf::Parameter Param;

        Param p;
        p.planar_ = param_.planar_;

        Inf inf(subGm,p);
        inf.infer();
        inf.arg(subArg);

        for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            res[vi] = subArg[ab[vi]];
        }
        const ValueType resultVal = subGm.evaluate(subArg);
        const ValueType projectedResultVal = gm_.evaluate(res);
        const std::vector<LabelType> & bestArg = valA < valB ? a : b;
        const ValueType bestProposalVal  =  valA < valB ? valA : valB;

        valRes = bestProposalVal < projectedResultVal ? bestProposalVal : projectedResultVal;
        if(projectedResultVal < bestProposalVal){
            //for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            //    res[vi] = subArg[ab[vi]];
            //}
        }
        else{
            for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
                res[vi] = bestArg[vi];
            }
        }
        return true;

    }

    bool doMoveMulticut(
        const MapType & accWeights,
        const std::vector<LabelType> & ab,
        const IndexType numNewVar,
        const std::vector<LabelType> & a,
        const std::vector<LabelType> & b,
        std::vector<LabelType> & res,
        const ValueType valA,
        const ValueType valB,
        ValueType & valRes
    ){
        // make the actual sub graphical model
        SubSpace subSpace(numNewVar, numNewVar);
        SubModel subGm(subSpace);

        // reserve space
        subGm. template reserveFunctions<PFunction>(accWeights.size());
        subGm.reserveFactors(accWeights.size());
        subGm.reserveFactorsVarialbeIndices(accWeights.size()*2);

        for(MapCIter i = accWeights.begin(); i!=accWeights.end(); ++i){
            const UInt64Type key    = i->first;
            const ValueType weight = i->second;

            const UInt64Type cVi1 = key/numNewVar;
            const UInt64Type cVi0 = key - cVi1*numNewVar;
            const UInt64Type vis[2] = {cVi0, cVi1};

            PFunction pf(numNewVar, numNewVar, 0.0, weight);
            subGm.addFactor(subGm.addFunction(pf), vis, vis+2);
        }

        std::vector<LabelType> subArg;

        //::cout<<"WITH MC\n";
        typedef Multicut<SubModel, Minimizer> Inf;
        typedef  typename  Inf::Parameter Param;
        Param p(0,0.0);
        Inf inf(subGm,p);
        inf.infer();
        inf.arg(subArg);

        for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            res[vi] = subArg[ab[vi]];
        }
        const ValueType resultVal = subGm.evaluate(subArg);
        const ValueType projectedResultVal = gm_.evaluate(res);
        const std::vector<LabelType> & bestArg = valA < valB ? a : b;
        const ValueType bestProposalVal  =  valA < valB ? valA : valB;

        valRes = bestProposalVal < projectedResultVal ? bestProposalVal : projectedResultVal;
        if(projectedResultVal < bestProposalVal){
            //for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            //    res[vi] = subArg[ab[vi]];
            //}
        }
        else{
            for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
                res[vi] = bestArg[vi];
            }
        }
        return true;
    }



    bool doMoveHierachicalClustering(
        const MapType & accWeights,
        const std::vector<LabelType> & ab,
        const IndexType numNewVar,
        const std::vector<LabelType> & a,
        const std::vector<LabelType> & b,
        std::vector<LabelType> & res,
        const ValueType valA,
        const ValueType valB,
        ValueType & valRes
    ){
        #ifndef NOVIGRA
        typedef vigra::AdjacencyListGraph Graph;
        typedef typename Graph::Edge Edge;
        typedef vigra::MergeGraphAdaptor< Graph > MergeGraph;
        typedef McClusterOp<GM,ACC> ClusterOp;
        typedef typename ClusterOp::Parameter ClusterOpParam;
        typedef vigra::HierarchicalClustering< ClusterOp > HC;
        typedef typename HC::Parameter HcParam;
        
        std::vector<ValueType> weights(accWeights.size(),0.0);

        Graph graph(numNewVar, accWeights.size());
        for(MapCIter i = accWeights.begin(); i!=accWeights.end(); ++i){
            const UInt64Type key    = i->first;
            const ValueType weight = i->second;

            const UInt64Type cVi1 = key/numNewVar;
            const UInt64Type cVi0 = key - cVi1*numNewVar;
            
            const Edge e = graph.addEdge(cVi0, cVi1);
            weights[graph.id(e)] = weight;
        }

        MergeGraph mg(graph);




        const ClusterOpParam clusterOpParam;
        ClusterOp clusterOp(graph, mg, clusterOpParam, weights);




        HcParam p;
        HC hc(clusterOp,p);

        //std::cout<<"start\n";
        hc.cluster();



        for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            res[vi] = hc.reprNodeId(ab[vi]);
        }

        const ValueType projectedResultVal = gm_.evaluate(res);
        const std::vector<LabelType> & bestArg = valA < valB ? a : b;
        const ValueType bestProposalVal  =  valA < valB ? valA : valB;

        valRes = bestProposalVal < projectedResultVal ? bestProposalVal : projectedResultVal;
        if(projectedResultVal < bestProposalVal){
            //for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
            //    res[vi] = subArg[ab[vi]];
            //}
        }
        else{
            for(IndexType vi=0; vi<gm_.numberOfVariables(); ++vi){
                res[vi] = bestArg[vi];
            }
        }
        return true;
        #else   
            throw RuntimeError("needs VIGRA");
            return fals
        #endif
    }

private:
    const GM & gm_;
    Parameter param_;
};





}


#endif /* OPENGM_PERMUTABLE_LABEL_FUSION_MOVER_HXX */
