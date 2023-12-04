#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  #pragma omp parallel for
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
  }
  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */

  double *solution_new = new double[numNodes];
  bool converged = false;
  int MaxThreadNumber = omp_get_max_threads();

  // Use [MaxThreadNumber][8] to avoid register false sharing
  double SumNoOutGoing_Threads[MaxThreadNumber][8];
  double GlobalDiff_Threads[MaxThreadNumber][8];
  double SumNoOutGoing ,  GlobalDiff;


  while(!converged){

    GlobalDiff = 0.0;
    SumNoOutGoing = 0.0;
    #pragma omp parallel for
    for(int i = 0 ; i < MaxThreadNumber ; i++){
      SumNoOutGoing_Threads[i][0] = 0.0;
      GlobalDiff_Threads[i][0] = 0.0;
    }

    #pragma omp parallel
    {
      int ThreadID = omp_get_thread_num();
      #pragma omp for schedule(dynamic, 512)
      for (int i=0; i < numNodes; i++) {
          // Vertex is typedef'ed to an int. Vertex* points into g.outgoing_edges[]
        int numOfOutgoing = outgoing_size(g, i);
        if(numOfOutgoing == 0 ) SumNoOutGoing_Threads[ThreadID][0] += solution[i];
      }   
    }

    for(int i = 0 ; i < MaxThreadNumber ; i++){
      SumNoOutGoing += SumNoOutGoing_Threads[i][0];
    }


    #pragma omp parallel
    {
      int ThreadID = omp_get_thread_num();
      #pragma omp for schedule(dynamic, 512)
      for (int i=0; i < numNodes; i++) {
          // Vertex is typedef'ed to an int. Vertex* points into g.outgoing_edges[]
          const Vertex* start = incoming_begin(g, i);
          const Vertex* end = incoming_end(g, i);
          double sum = 0.0;
          for (const Vertex* v=start; v!=end; v++){
            int numOfOutgoing = outgoing_size(g, *v);
            sum += (solution[*v]/numOfOutgoing);
          }
          solution_new[i] = (1.0 - damping) / numNodes + damping * sum;
          solution_new[i] += damping * SumNoOutGoing / numNodes;

          GlobalDiff_Threads[ThreadID][0] += fabs(solution_new[i] -solution[i]);
      }    
      
    }

    for (int i = 0;i < MaxThreadNumber ;i++){
      GlobalDiff += GlobalDiff_Threads[i][0];
    } 
    converged = (GlobalDiff < convergence);

    #pragma omp parallel for
    for (int i = 0 ; i < numNodes ; i++ ) {
      solution[i] = solution_new[i];
    }

  }

}
