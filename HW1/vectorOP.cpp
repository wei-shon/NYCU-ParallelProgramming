#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    __pp_mask mask = _pp_init_ones();
    // if( (i + 1)*VECTOR_WIDTH > N){
    //   for(int j = 0 ; j < VECTOR_WIDTH ; j++){
    //     if(i + j >= N) mask.value[j] = false;
    //   }
    // }
    // Initial value and exponent 
    __pp_vec_float result ;

    __pp_vec_float value;
    _pp_vload_float(value , values+ i , mask);
    _pp_vmove_float(result, value , mask);

    __pp_vec_int exponent ;
    _pp_vload_int(exponent , exponents+ i , mask);

    for(int j = 0 ; j < VECTOR_WIDTH ; j++){
      if(exponent.value[j] == 0 ){
        mask.value[j] = false;
        result.value[j] = 1.f;
      } 
    }

    int count = 1;
    while(_pp_cntbits(mask)){
      for(int j = 0 ; j < VECTOR_WIDTH ; j++){
        if(exponent.value[j] <= count) mask.value[j] = false;
        if (result.value[j] > 9.999999f) result.value[j] = 9.999999f;
      }
      _pp_vmult_float(result, result, value, mask);
      // __pp_mask mask = _pp_init_ones(i);
      count++;
    }
    mask = _pp_init_ones();
    if( (i + 1)*VECTOR_WIDTH > N){
      for(int j = 0 ; j < VECTOR_WIDTH ; j++){
        if(i + j >= N) mask.value[j] = false;
      }
    }
    _pp_vstore_float(output+ i, result, mask);
  } 
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here 
  //

  float sum = 0;
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    __pp_mask mask = _pp_init_ones();
    __pp_vec_float result_interleave;
    __pp_vec_float result;
    __pp_vec_float value;
    _pp_vload_float(value , values+ i , mask);
    _pp_hadd_float(result , value);
    _pp_interleave_float(result_interleave , result);
    for(int j = 0 ; j < VECTOR_WIDTH/2 ; j++){
      sum+=result_interleave.value[j];
    }
  }

  return sum;
}