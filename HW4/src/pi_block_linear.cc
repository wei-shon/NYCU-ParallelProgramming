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
    srand( time(NULL) );

    // TODO: init MPI
    MPI_Comm_size(MPI_COMM_WORLD , &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int number_in_circle = 0;
    int recieve_circle = 0;
    MPI_Status status;/* return status for   */ 
    int tag=0; /* tag for messages    */
    int min = -1;
    int max = 1;
    // each toss number
    int each_toss = tosses / world_size;

    for (long long int toss = 0; toss < each_toss; toss ++) {
        double x = (max - min) * (rand() / (RAND_MAX + 1.0)) + min;
        double y = (max - min) * (rand() / (RAND_MAX + 1.0)) + min;
        // cout<<x<<" "<<y<<endl;
        double distance_squared = x * x + y * y;
        if ( distance_squared <= 1)
                recieve_circle++;
    }

    if (world_rank > 0)
    {
        // TODO: handle workers
        int dest = 0;  /* rank of 0 is receiver    */
        MPI_Send(&recieve_circle, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);

    }
    else if (world_rank == 0)
    {
        // TODO: master
        number_in_circle+=recieve_circle;
        for (int source = 1; source < world_size; source++ ) {
            MPI_Recv(&recieve_circle, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status); 
            number_in_circle+=recieve_circle;
        }
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
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
