#include <CL/cl.h>
#include <iostream>
#include <vector>

#include "ocl_context.h"
#include "lz_context.h"

void simple_interop()
{
    size_t elemCount = 1024 * 1024;
    std::vector<uint32_t> initBuf(elemCount, 0);
    for (size_t i = 0; i < elemCount; i++)
        initBuf[i] = (i % 1024);

    // initialize opencl
    oclContext oclctx;
    oclctx.init(0);

    // initialize level-zero
    lzContext lzctx;
    lzctx.initZe(0);

    // create opencl buffer and derive dma-buf handle from it
    cl_mem clBuffer = oclctx.createBuffer(elemCount * sizeof(uint32_t), initBuf);
    oclctx.printBuffer(clBuffer);
    uint64_t handle = oclctx.deriveHandle(clBuffer);

    // create level-zero device memory from the handle
    void *lzptr = lzctx.createFromHandle(handle, elemCount * sizeof(uint32_t));
    lzctx.printBuffer(lzptr);

    oclctx.freeBuffer(clBuffer);
}

int test_base(int argc, char **argv)
{
    // simple_interop();

    size_t elemCount = 512;
    std::vector<uint32_t> initBuf(elemCount, 0);
    for (size_t i = 0; i < elemCount; i++)
        initBuf[i] = (i % 1024);

    // initialize two opencl contexts, oclctx0 on GPU0 and oclctx1 on GPU1
    oclContext oclctx0, oclctx1;
    oclctx0.init(0);
    oclctx1.init(1);

    // initialize two level-zero contexts, lzctx0 on GPU0 and lzctx1 on GPU1
    lzContext lzctx0, lzctx1;
    lzctx0.initZe(0);
    lzctx1.initZe(1);

    // create two opencl buffers on the device memory of GPU0 and GPU1 respectively
    cl_mem clbuf0 = oclctx0.createBuffer(elemCount * sizeof(uint32_t), initBuf);
    cl_mem clbuf1 = oclctx1.createBuffer(elemCount * sizeof(uint32_t), initBuf);
    oclctx0.printBuffer(clbuf0);
    oclctx1.printBuffer(clbuf1);

    // derive the dma-buf handles from opencl buffers
    uint64_t handle0 = oclctx0.deriveHandle(clbuf0);
    uint64_t handle1 = oclctx1.deriveHandle(clbuf1);

    // create two level-zero device memory on GPU0/GPU1 based on the dma-buf handles
    void *lzptr0 = lzctx0.createFromHandle(handle0, elemCount * sizeof(uint32_t));
    void *lzptr1 = lzctx1.createFromHandle(handle1, elemCount * sizeof(uint32_t));
    lzctx0.printBuffer(lzptr0);
    lzctx1.printBuffer(lzptr1);

    // run p2p data transfer kernel: GPU0 read data from GPU1
    lzctx0.runKernel("../../lz_p2p/test_kernel_dg2.spv", "local_read_from_remote", lzptr1, lzptr0, elemCount);

    // run p2p data transfer kernel: GPU0 write data to GPU1
    lzctx0.runKernel("../../lz_p2p/test_kernel_dg2.spv", "local_write_to_remote", lzptr1, lzptr0, elemCount);

    // print the content of the original opencl buffers, the data was changed after above level-zero kernel execution.
    oclctx0.printBuffer(clbuf0);
    oclctx1.printBuffer(clbuf1);

    oclctx0.freeBuffer(clbuf0);
    oclctx1.freeBuffer(clbuf1);

    return 0;
}

int test_matrix_add(int argc, char **argv)
{
    cl_int err;
    size_t size = 1024;
    size_t elemCount = size * size;
    std::vector<uint32_t> A(size * size, 1); // Initialize matrix A for device 0
    std::vector<uint32_t> B(size * size, 2); // Initialize matrix B for device 0
    std::vector<uint32_t> D(size * size, 4); // Initialize matrix D for device 1

    // initialize two opencl contexts, oclctx0 on GPU0 and oclctx1 on GPU1
    oclContext oclctx0, oclctx1;
    oclctx0.init(0);
    oclctx1.init(1);

    // initialize two level-zero contexts, lzctx0 on GPU0 and lzctx1 on GPU1
    lzContext lzctx0, lzctx1;
    lzctx0.initZe(0);
    lzctx1.initZe(1);

    // create 3 opencl buffers on the device memory of GPU0 - A + B = C
    cl_mem clbufA = oclctx0.createBuffer(elemCount * sizeof(uint32_t), A);
    cl_mem clbufB = oclctx0.createBuffer(elemCount * sizeof(uint32_t), B);
    cl_mem clbufC = clCreateBuffer(oclctx0.context(), CL_MEM_READ_WRITE, sizeof(uint32_t) * size * size, nullptr, &err);
    //cl_mem clbufC = oclctx0.createBuffer(elemCount * sizeof(uint32_t), B);
    CHECK_OCL_ERROR_EXIT(err, "clCreateBuffer");
    oclctx0.printBuffer(clbufA);
    oclctx0.printBuffer(clbufB);

    // Define kernel for matrix addition
    char* kernelSource = R"(
        __kernel void matrix_add(__global const int* X, __global const int* Y, __global int* Z) {
            int index = get_global_id(0);
            Z[index] = X[index] + Y[index];
            // printf("%d + %d = %d\n", X[index],Y[index],Z[index]);
        }
    )";
    // GPU0: A + B = C
    oclctx0.runKernel(kernelSource, "matrix_add", clbufA, clbufB, clbufC, elemCount);
    oclctx0.printBuffer(clbufC);

    // create 2 opencl buffers on the device memory of GPU1: C + D = E
    cl_mem clbufD = oclctx1.createBuffer(elemCount * sizeof(uint32_t), D);
    cl_mem clbufE = clCreateBuffer(oclctx1.context(), CL_MEM_READ_WRITE, sizeof(uint32_t) * size * size, nullptr, &err);
    //cl_mem clbufE = oclctx1.createBuffer(elemCount * sizeof(uint32_t), D);
    CHECK_OCL_ERROR_EXIT(err, "clCreateBuffer");
    oclctx1.printBuffer(clbufD);
    oclctx1.printBuffer(clbufE);

    // derive the dma-buf handles from opencl buffers
    uint64_t handleC = oclctx0.deriveHandle(clbufC);
    uint64_t handleD = oclctx1.deriveHandle(clbufD);
    uint64_t handleE = oclctx1.deriveHandle(clbufE);

    // create two level-zero device memory on GPU0 based on the dma-buf handles
    void *lzptrC = lzctx0.createFromHandle(handleC, elemCount * sizeof(uint32_t));
    void *lzptrD = lzctx1.createFromHandle(handleD, elemCount * sizeof(uint32_t));
    void *lzptrE = lzctx1.createFromHandle(handleE, elemCount * sizeof(uint32_t));
    lzctx0.printBuffer(lzptrC);
    lzctx1.printBuffer(lzptrD);
    lzctx1.printBuffer(lzptrE);

    std::cout << std::endl;
    // run p2p data transfer kernel: GPU1 read data from GPU0 and add data in GPU1
    std::cout << "GPU0 read remote data to add into local buffer ..." << std::endl;
    lzctx0.runKernel("../../interop/test_kernel_dg2.spv", "remote_matrix_add", lzptrD, lzptrD, lzptrC, elemCount);
    // print the content of the original opencl buffers, the data was changed after above level-zero kernel execution.
    lzctx0.printBuffer(lzptrC); //8888888888
    std::cout << std::endl;

    std::cout << std::endl;
    // run p2p data transfer kernel: GPU1 read data from GPU0 and add data in GPU1
    std::cout << "GPU1 read remote data and local data then add into GPU1 ..." << std::endl;
    lzctx0.runKernel("../../interop/test_kernel_dg2.spv", "remote_matrix_add", lzptrD, lzptrC, lzptrC, elemCount);
    // print the content of the original opencl buffers, the data was changed after above level-zero kernel execution.
    lzctx0.printBuffer(lzptrC); //12-12-12-12
    std::cout << std::endl;

    std::cout << std::endl;
    // run p2p data transfer kernel: GPU1 read data from GPU0 and add data in GPU1
    std::cout << "GPU1 read remote data and local data then add into GPU1 ..." << std::endl;
    lzctx0.runKernel("../../interop/test_kernel_dg2.spv", "remote_matrix_add", lzptrD, lzptrC, lzptrC, elemCount);
    // print the content of the original opencl buffers, the data was changed after above level-zero kernel execution.
    lzctx0.printBuffer(lzptrC); //12-12-12-12
    std::cout << std::endl;

    // run p2p data transfer kernel: GPU0 write data to GPU1
    std::cout << "GPU1 local read/write data into GPU1 ..." << std::endl;
    lzctx1.runKernel("../../interop/test_kernel_dg2.spv", "remote_matrix_add", lzptrD, lzptrD, lzptrE, elemCount);
    // print the content of the original opencl buffers, the data was changed after above level-zero kernel execution.
    lzctx1.printBuffer(lzptrE); //24-24-24-24
    std::cout << std::endl;

    // run p2p data transfer kernel: GPU0 write data to GPU1
    std::cout << "GPU0 remote read/write data into GPU1 ..." << std::endl;
    lzctx0.runKernel("../../interop/test_kernel_dg2.spv", "remote_matrix_add", lzptrC, lzptrD, lzptrE, elemCount);
    // print the content of the original opencl buffers, the data was changed after above level-zero kernel execution.
    lzctx1.printBuffer(lzptrE); //24-24-24-24
    std::cout << std::endl;
#if 0
    // run p2p data transfer kernel: GPU0 read data drom GPU1
    std::cout << "GPU0 remote 2x read data from GPU1 ..." << std::endl;
    lzctx0.runKernel("../../interop/test_kernel_dg2.spv", "remote_matrix_add", lzptrE, lzptrD, lzptrC, elemCount);
    // print the content of the original opencl buffers, the data was changed after above level-zero kernel execution.
    lzctx0.printBuffer(lzptrC);//1111111111111
    std::cout << std::endl;

    // run p2p data transfer kernel: GPU1 write data to GPU0
    std::cout << "GPU1 remote 2x write data into GPU0 ..." << std::endl;
    lzctx1.runKernel("../../interop/test_kernel_dg2.spv", "remote_matrix_add", lzptrE, lzptrD, lzptrC, elemCount);
    // print the content of the original opencl buffers, the data was changed after above level-zero kernel execution.
    lzctx0.printBuffer(lzptrC);//1111111111111
    std::cout << std::endl;

    // run p2p data transfer kernel: GPU1 write/read data in GPU1
    std::cout << "GPU1 read/write data into GPU1 ..." << std::endl;
    lzctx1.runKernel("../../interop/test_kernel_dg2.spv", "remote_matrix_add", lzptrE, lzptrD, lzptrE, elemCount);
    // print the content of the original opencl buffers, the data was changed after above level-zero kernel execution.
    lzctx0.printBuffer(lzptrC);//1111111
#endif
    return 0;
}

int test_ocl_l0_sync(int argc, char **argv)
{
    cl_int err;
    size_t size = 1024;
    size_t elemCount = size * size;
    std::vector<uint32_t> A(size * size, 1); // Initialize matrix A for device 0
    std::vector<uint32_t> B(size * size, 2); // Initialize matrix B for device 0
    std::vector<uint32_t> D(size * size, 4); // Initialize matrix D for device 1

    // initialize two opencl contexts, oclctx0 on GPU0 and oclctx1 on GPU1
    oclContext oclctx0, oclctx1;
    oclctx0.init(0);
    oclctx1.init(1);

    // initialize two level-zero contexts, lzctx0 on GPU0 and lzctx1 on GPU1
    lzContext lzctx0, lzctx1;
    lzctx0.initZe(0);
    lzctx1.initZe(1);

    // create 3 opencl buffers on the device memory of GPU0 - A + B = C
    cl_mem clbufA = oclctx0.createBuffer(elemCount * sizeof(uint32_t), A);
    cl_mem clbufB = oclctx0.createBuffer(elemCount * sizeof(uint32_t), B);
    cl_mem clbufC = clCreateBuffer(oclctx0.context(), CL_MEM_READ_WRITE, sizeof(uint32_t) * size * size, nullptr, &err);
    //cl_mem clbufC = oclctx0.createBuffer(elemCount * sizeof(uint32_t), B);
    CHECK_OCL_ERROR_EXIT(err, "clCreateBuffer");
    oclctx0.printBuffer(clbufA);
    oclctx0.printBuffer(clbufB);

    // Define kernel for matrix addition
    char* kernelSource = R"(
        __kernel void matrix_add(__global const int* X, __global const int* Y, __global int* Z) {
            int index = get_global_id(0);
            Z[index] = X[index] + Y[index];
            // printf("%d + %d = %d\n", X[index],Y[index],Z[index]);
        }
    )";
    // GPU0: A + B = C
    oclctx0.runKernel(kernelSource, "matrix_add", clbufA, clbufB, clbufC, elemCount);
    oclctx0.runKernel(kernelSource, "matrix_add", clbufA, clbufB, clbufC, elemCount);
    oclctx0.runKernel(kernelSource, "matrix_add", clbufA, clbufB, clbufC, elemCount);
    oclctx0.printBuffer(clbufC);

    // create 2 opencl buffers on the device memory of GPU1: C + D = E
    cl_mem clbufD = oclctx1.createBuffer(elemCount * sizeof(uint32_t), D);
    cl_mem clbufE = clCreateBuffer(oclctx1.context(), CL_MEM_READ_WRITE, sizeof(uint32_t) * size * size, nullptr, &err);
    //cl_mem clbufE = oclctx1.createBuffer(elemCount * sizeof(uint32_t), D);
    CHECK_OCL_ERROR_EXIT(err, "clCreateBuffer");
    oclctx1.printBuffer(clbufD);
    oclctx1.printBuffer(clbufE);

    // derive the dma-buf handles from opencl buffers
    uint64_t handleC = oclctx0.deriveHandle(clbufC);
    uint64_t handleD = oclctx1.deriveHandle(clbufD);
    uint64_t handleE = oclctx1.deriveHandle(clbufE);

    // create two level-zero device memory on GPU0 based on the dma-buf handles
    void *lzptrC = lzctx0.createFromHandle(handleC, elemCount * sizeof(uint32_t));
    void *lzptrD = lzctx1.createFromHandle(handleD, elemCount * sizeof(uint32_t));
    void *lzptrE = lzctx1.createFromHandle(handleE, elemCount * sizeof(uint32_t));
    lzctx0.printBuffer(lzptrC);
    lzctx1.printBuffer(lzptrD);
    lzctx1.printBuffer(lzptrE);

    std::cout << std::endl;
    // run p2p data transfer kernel: GPU0 write data to GPU1
    std::cout << "GPU0 remote write data into GPU0 ..." << std::endl;
    lzctx0.runKernel("../../interop/test_kernel_dg2.spv", "remote_matrix_add", lzptrC, lzptrC, lzptrE, elemCount);
    lzctx0.runKernel("../../interop/test_kernel_dg2.spv", "remote_matrix_add", lzptrC, lzptrC, lzptrE, elemCount);
    // print the content of the original opencl buffers, the data was changed after above level-zero kernel execution.
    lzctx0.printBuffer(lzptrE);
    std::cout << std::endl;

    // GPU0: A + B = C
    oclctx0.runKernel(kernelSource, "matrix_add", clbufA, clbufB, clbufC, elemCount);
    oclctx0.runKernel(kernelSource, "matrix_add", clbufA, clbufB, clbufC, elemCount);
    oclctx0.runKernel(kernelSource, "matrix_add", clbufA, clbufB, clbufC, elemCount);

    // run p2p data transfer kernel: GPU0 write data to GPU1
    std::cout << "GPU0 remote write data into GPU0 ..." << std::endl;
    lzctx0.runKernel("../../interop/test_kernel_dg2.spv", "remote_matrix_add", lzptrC, lzptrC, lzptrE, elemCount);
    lzctx0.runKernel("../../interop/test_kernel_dg2.spv", "remote_matrix_add", lzptrC, lzptrC, lzptrE, elemCount);
    // print the content of the original opencl buffers, the data was changed after above level-zero kernel execution.
    lzctx1.printBuffer(lzptrE);
    std::cout << std::endl;

    return 0;
}

int main(int argc, char **argv) {
    //test_base(argc, argv);
    //test_matrix_add(argc, argv);
    test_ocl_l0_sync(argc,argv);
}
