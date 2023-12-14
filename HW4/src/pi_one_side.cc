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

    MPI_Win win;

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
    
    int *eachCircle;
    if (world_rank == 0)
    {
        // Master
        MPI_Alloc_mem(world_size * sizeof(int), MPI_INFO_NULL, &eachCircle);
        // initial the eachCircle
        *eachCircle = 0;
        MPI_Win_create(eachCircle, world_size * sizeof(int), sizeof(int), MPI_INFO_NULL,
          MPI_COMM_WORLD, &win);
    }
    else
    {
        // Workers
        // Worker processes do not expose memory in the window
       MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD,&win);

       // Register with the master
       MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
       //since we want to save circle into rank = 0
       int target_disp = 0;
       MPI_Accumulate(&number_in_circle, 1, MPI_INT, 0, target_disp, 1, MPI_INT, MPI_SUM, win);
       MPI_Win_unlock(0, win);
    }

    // Synchronize all processes before one-sided communication
    MPI_Win_fence(0, win);
    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        // I need to add other process circle
        number_in_circle += *eachCircle;
        pi_result = 4 * (static_cast<double>(number_in_circle) / tosses);
        // Free the allocated memory
        MPI_Free_mem(eachCircle);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}