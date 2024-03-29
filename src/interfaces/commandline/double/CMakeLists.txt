#add_definitions(-DTRWS_DEBUG_OUTPUT)

 if(WITH_HDF5)
    add_executable(opengm_max_prod opengm_max_prod.cxx ${headers})
    add_executable(opengm_min_sum opengm_min_sum.cxx ${headers})
    add_executable(opengm_min_sum_small opengm_min_sum_small.cxx ${headers})
    if(WITH_LIBDAI)
      add_executable(opengm_min_sum_libdai opengm_min_sum_libdai.cxx ${headers})
      target_link_libraries(opengm_min_sum_libdai ${LIBDAI_LIBRARY})
      target_link_libraries(opengm_min_sum_libdai ${GMP_LIBRARY})
      target_link_libraries(opengm_min_sum_libdai ${GMPXX_LIBRARY})
      target_link_libraries(opengm_min_sum_libdai ${HDF5_LIBRARIES})
      if(LINK_RT)
         find_library(RT rt)
         target_link_libraries(opengm_min_sum_libdai rt)
      endif()
    endif()

    if(WITH_OPENMP)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
      SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    endif()

    target_link_libraries(opengm_max_prod ${HDF5_LIBRARIES})
    target_link_libraries(opengm_min_sum ${HDF5_LIBRARIES})
    target_link_libraries(opengm_min_sum_small ${HDF5_LIBRARIES})
    MESSAGE(STATUS  ${LINK_RT} )  
    if(LINK_RT)
        find_library(RT rt)
        target_link_libraries(opengm_max_prod rt)
        target_link_libraries(opengm_min_sum rt)  
        target_link_libraries(opengm_min_sum_small rt)
    endif(LINK_RT) 

    if(WITH_FASTPD)
      target_link_libraries(opengm_min_sum external-library-fastpd)   
    endif(WITH_FASTPD)

    if(WITH_GCO)
      target_link_libraries(opengm_min_sum external-library-gco)
    endif(WITH_GCO)

    if(WITH_MAXFLOW)
        target_link_libraries(opengm_max_prod external-library-maxflow)
        target_link_libraries(opengm_min_sum external-library-maxflow)
        target_link_libraries(opengm_min_sum_small external-library-maxflow)
    endif(WITH_MAXFLOW)
    if(WITH_CPLEX)
        if(WIN32)
	       target_link_libraries(opengm_max_prod wsock32.lib ${CPLEX_LIBRARIES})
	       target_link_libraries(opengm_min_sum  wsock32.lib  ${CPLEX_LIBRARIES})
      	       target_link_libraries(opengm_min_sum_small  wsock32.lib  ${CPLEX_LIBRARIES})
        else()
	       target_link_libraries(opengm_max_prod ${CMAKE_THREAD_LIBS_INIT}  ${CPLEX_LIBRARIES} )
	       target_link_libraries(opengm_min_sum  ${CMAKE_THREAD_LIBS_INIT}  ${CPLEX_LIBRARIES} )
      	       target_link_libraries(opengm_min_sum_small ${CMAKE_THREAD_LIBS_INIT}  ${CPLEX_LIBRARIES} )
        endif(WIN32)
    endif(WITH_CPLEX)

    if(WITH_QPBO)
           target_link_libraries(opengm_max_prod external-library-qpbo )
           target_link_libraries(opengm_min_sum  external-library-qpbo )
           target_link_libraries(opengm_min_sum_small  external-library-qpbo )
    endif(WITH_QPBO)

    if(WITH_CONICBUNDLE)
      target_link_libraries(opengm_min_sum ${CONICBUNDLE_LIBRARY})
      target_link_libraries(opengm_min_sum_small ${CONICBUNDLE_LIBRARY})
    endif(WITH_CONICBUNDLE)
    
    if(WITH_MRF)
      target_link_libraries(opengm_min_sum external-library-mrf)
      target_link_libraries(opengm_min_sum_small external-library-mrf)
    endif(WITH_MRF)  

    if(WITH_GRANTE)
      target_link_libraries(opengm_min_sum external-library-grante)
      target_link_libraries(opengm_min_sum ${Boost_LIBRARIES})
    endif(WITH_GRANTE)
    
    if(WITH_DAOOPT)
       target_link_libraries(opengm_min_sum external-library-daoopt)
    endif(WITH_DAOOPT)
    
    if(WITH_MPLP)
       target_link_libraries(opengm_min_sum external-library-mplp)
    endif(WITH_MPLP)

    if(WITH_AD3)
       target_link_libraries(opengm_min_sum external-library-ad3)
       target_link_libraries(opengm_min_sum_small external-library-ad3)
    endif(WITH_AD3)


    if(WITH_GUROBI)
      target_link_libraries(opengm_min_sum ${CMAKE_THREAD_LIBS_INIT} 
        ${GUROBI_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
      )
    endif()


    if(WITH_ISINF)
      target_link_libraries(opengm_min_sum external-library-isinf-shared)
    endif(WITH_ISINF)




    INSTALL_TARGETS(/bin opengm_max_prod) 
    INSTALL_TARGETS(/bin opengm_min_sum)
    INSTALL_TARGETS(/bin opengm_min_sum_small)
    if(WITH_LIBDAI)
      INSTALL_TARGETS(/bin opengm_min_sum_libdai)
    endif()
  endif(WITH_HDF5)
