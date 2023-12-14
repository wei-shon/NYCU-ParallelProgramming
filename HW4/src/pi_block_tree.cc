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
    int recieve_circle = 0;
    int min = -1;
    int max = 1;
    int each_toss = tosses / world_size;

    for (long long int toss = 0; toss < each_toss; toss ++) {
        double x = (max - min) * (rand() / (RAND_MAX + 1.0)) + min;
        double y = (max - min) * (rand() / (RAND_MAX + 1.0)) + min;
        double distance_squared = x * x + y * y;
        if ( distance_squared <= 1)
                number_in_circle++;
    }
    // TODO: binary tree redunction
    int step = 1;
    while (step < world_size) {
        if(world_rank % (2 * step) != 0 ){
            int destination = world_rank - step;
            MPI_Send(&number_in_circle, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
            break;  // Exit the loop for processes that have sent their value
        }
        else{
            int source = world_rank + step;
            if (source < world_size) {
                MPI_Recv(&recieve_circle, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                number_in_circle += recieve_circle;
            }
        } 
        step *= 2;
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
