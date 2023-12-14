hello:
	mpicxx ./hello.cc -o mpi_hello
	mpirun -np 8 --hostfile hosts mpi_hellomak

linear:
	mpicxx pi_block_linear.cc -o pi_block_linear
	mpirun -np 4 --hostfile hosts pi_block_linear 1000000000

tree:
	mpicxx pi_block_tree.cc -o pi_block_tree
	mpirun -np 8 --hostfile hosts pi_block_tree 1000000000

nonblock:
	mpicxx pi_nonblock_linear.cc -o pi_nonblock_linear
	mpirun -np 4 --hostfile hosts pi_nonblock_linear 1000000000

gather:
	mpicxx pi_gather.cc -o pi_gather
	mpirun -np 4 --hostfile hosts pi_gather 1000000000

reduce:
	mpicxx pi_reduce.cc -o pi_reduce
	mpirun -np 4 --hostfile hosts pi_reduce 1000000000

oneSide:
	mpicxx pi_one_side.cc -o pi_one_side
	mpirun -np 4 --hostfile hosts pi_one_side 1000000000

install:
	parallel-scp -h host.txt -r ~/HW4 ~