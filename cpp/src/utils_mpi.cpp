#include "utils_mpi.h"

unsigned int EvaluateBestSplitSize(unsigned int v_size, int world_size)
{
  double v_sub_size_perfect = (double)v_size / (double)world_size;
  unsigned int v_sub_size = floor(v_sub_size_perfect);
  if ((double)v_sub_size != v_sub_size_perfect) v_sub_size += 1;
  return(v_sub_size);
}

