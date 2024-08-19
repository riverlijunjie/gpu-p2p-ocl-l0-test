
kernel void vector_add(global int *src1, global int *src2) 
{
  const int id = get_global_id(0);
  src1[id] = src1[id] + src2[id];
}

kernel void local_read_from_remote(global int *src1, global int *src2) 
{
  const int id = get_global_id(0);
  src1[id] = src2[id] * 3;
}

kernel void local_write_to_remote(global int *src1, global int *src2) 
{
  const int id = get_global_id(0);
  src2[id] = src1[id] * 5;
}
kernel void remote_matrix_add(global int *src1, global int *src2, global int *dst) 
{
  const int id = get_global_id(0);
  dst[id] = src1[id] + src2[id];
}