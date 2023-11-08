#include <iostream>
#include "test.h"
#include "fasttime.h"

void test2(float *__restrict a, float *__restrict b, float *__restrict c, int N, int count)
{
  __builtin_assume(N == 1024);
  
  double elapsedf = 0;

  for(int k = 0 ; k < count ; k++){ 
    a = (float *)__builtin_assume_aligned(a, 16);
    b = (float *)__builtin_assume_aligned(b, 16);
    c = (float *)__builtin_assume_aligned(c, 16);
    fasttime_t time1 = gettime();
    for (int i = 0; i < I; i++)
    {
      for (int j = 0; j < N; j++)
      {
        /* max() */
        if (b[j] > a[j]) c[j] = b[j];
        else c[j] = a[j];
      }
    }
    fasttime_t time2 = gettime();
    elapsedf += tdiff(time1, time2);
  }
  elapsedf = elapsedf / count;
  std::cout << "Elapsed execution time of the loop in test2():\n"
            << elapsedf << "sec (N: " << N << ", I: " << I << ")\n";
}
