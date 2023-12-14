all:
	mpicxx main.cc matrix.cc -o matmul 

test:
	mpirun -np 4 --hostfile hosts matmul < /home/311552024/data-set/data1_1 > output.txt

install:
	parallel-scp -h host.txt -r ~/HW4 ~