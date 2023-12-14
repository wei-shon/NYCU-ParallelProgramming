#include <stdio.h>
#include <mpi.h>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */ 
#define FROM_WORKER 2          /* setting a message type */


// Read size of matrix_a and matrix_b (n, m, l) and whole data of matrixes from stdin
//
// n_ptr:     pointer to n
// m_ptr:     pointer to m
// l_ptr:     pointer to l
// a_mat_ptr: pointer to matrix a (a should be a continuous memory space for placing n * m elements of int)
// b_mat_ptr: pointer to matrix b (b should be a continuous memory space for placing m * l elements of int)
void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr){

    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD , &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank == 0) {
        scanf("%d %d %d", n_ptr, m_ptr, l_ptr);
        *a_mat_ptr = (int*)malloc(sizeof(int) * (*n_ptr) * (*m_ptr));
        *b_mat_ptr = (int*)malloc(sizeof(int) * (*m_ptr) * (*l_ptr));

        for(int i = 0 ; i < (*n_ptr) ; i++)
            for(int j = 0 ; j < (*m_ptr) ; j++)
                scanf("%d", *a_mat_ptr + i * (*m_ptr) + j );
        
        for(int i = 0 ; i < (*m_ptr) ; i++)
            for(int j = 0 ; j < (*l_ptr) ; j++)
                scanf("%d", *b_mat_ptr + i * (*l_ptr) + j );
                
        // for(int i = 0 ; i < (*n_ptr) ; i++){
        //     for(int j = 0 ; j < (*m_ptr) ; j++){
        //         printf("%d ", *(*a_mat_ptr + i * (*m_ptr) + j) );
        //     }
        //     printf("\n");
        // }
            
        // for(int i = 0 ; i < (*m_ptr) ; i++){
        //     for(int j = 0 ; j < (*l_ptr) ; j++){
        //         printf("%d ", *(*b_mat_ptr + i * (*l_ptr) + j) );
        //     }
        //     printf("\n");
        // }  
    }
}

// Just matrix multiplication (your should output the result in this function)
// 
// n:     row number of matrix a
// m:     col number of matrix a / row number of matrix b
// l:     col number of matrix b
// a_mat: a continuous memory placing n * m elements of int
// b_mat: a continuous memory placing m * l elements of int
void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat){

    MPI_Status status;
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD , &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int averow, extra, offset, mtype, rows, source;
    int numworkers = world_size - 1;
    if (world_rank == MASTER) {
        averow = n/numworkers;
        extra = n%numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        int *c_mat = (int*)malloc(sizeof(int) * n * l);

        for (int dest = 1; dest <= numworkers; dest++)
        {
            MPI_Send(&n, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&m, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&l, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);

            rows = (dest <= extra) ? averow+1 : averow;
            // printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&a_mat[offset * m], rows*m, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&b_mat[0], m*l, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            offset = offset + rows; 
        }
        mtype = FROM_WORKER;
        for (int i = 1; i <= numworkers; i++)
        {
            // printf("Start to Recieve!\n");
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&c_mat[offset * l], rows*l, MPI_INT, source, mtype, MPI_COMM_WORLD, &status); 
            // printf("Received results from task %d\n",source);
        } 

        // printf("******************************************************\n"); 
        // printf("Result Matrix:\n");  
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < l; j++)
                printf("%d ", c_mat[i * l + j]); 
            printf("\n");
        } 
        // printf("\n******************************************************\n"); 
        // printf ("Done.\n");
    }

    if (world_rank > MASTER) {
        mtype = FROM_MASTER;

        int nCopy, mCopy, lCopy;
        MPI_Recv(&nCopy, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&mCopy, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&lCopy, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        // printf("%d rank is %d %d %d!\n",world_rank, nCopy,mCopy,lCopy);
        int *a = (int*)malloc(sizeof(int) * nCopy * mCopy);
        int *b = (int*)malloc(sizeof(int) * mCopy * lCopy);
        int *c = (int*)malloc(sizeof(int) * nCopy * lCopy);
        // printf("%d Start to Recieve!\n", world_rank);

        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&a[0], rows * mCopy, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&b[0], mCopy * lCopy, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status); 
        for (int k = 0; k<lCopy; k++){
            for (int i = 0; i<rows; i++) {
                c[i * lCopy + k] = 0.0;
                for (int j = 0; j<mCopy; j++)
                    c[i * lCopy + k] = c[i * lCopy + k] + a[i * mCopy + j] * b[j * lCopy + k];
            }
        }
        

        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&c[0], rows * lCopy, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);

        free(a);
        free(b);
        free(c);
    } 
}

// Remember to release your allocated memory
void destruct_matrices(int *a_mat, int *b_mat){
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank == MASTER){
        free(a_mat);
        free(b_mat);
    }
    

}