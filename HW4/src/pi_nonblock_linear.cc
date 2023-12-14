#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD , &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    srand( time(NULL) );

    int number_in_circle = 0;
    int min = -1;
    int max = 1;
    int each_toss = tosses / world_size;

    for (long long int toss = 0; toss < each_toss; toss ++) {
        double x = (max - min) * (rand() / (RAND_MAX + 1.0)) + min;
        double y = (max - min) * (rand() / (RAND_MAX + 1.0)) + min;
        // cout<<x<<" "<<y<<endl;
        double distance_squared = x * x + y * y;
        if ( distance_squared <= 1)
                number_in_circle++;
    }
    
    if (world_rank > 0)
    {
        // TODO: MPI workers
        int dest = 0;  /* rank of 0 is receiver    */
        MPI_Request req;
        MPI_Isend(&number_in_circle, 1, MPI_INT, dest, 0, MPI_COMM_WORLD, &req);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request requests[world_size -1 ];
        MPI_Status status[world_size - 1];
        int each_circle[world_size];
        for (int source = 1; source < world_size; source++ ) 
            MPI_Irecv(&(each_circle[source]), 1, MPI_INT, source, 0, MPI_COMM_WORLD, &(requests[source - 1]));

        MPI_Waitall(world_size-1, requests, status);
        for(int i = 1 ; i < world_size; i++) number_in_circle+=each_circle[i];
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 * (static_cast<double>(number_in_circle) / tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
