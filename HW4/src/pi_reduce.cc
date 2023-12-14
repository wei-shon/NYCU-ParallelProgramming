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
                recieve_circle++;
    }
    // TODO: use MPI_Reduce
    MPI_Reduce(&recieve_circle, &number_in_circle, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

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
