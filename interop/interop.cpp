
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <thread>
#include <functional>

#include "ocl_context.h"
#include "lz_context.h"

char read_kernel_code[] = " \
kernel void read_from_remote(global int *src1, global int *src2) \
{ \
  const int id = get_global_id(0); \
  src1[id] = src2[id] * 3; \
} \
";

char write_kernel_code[] = " \
kernel void write_to_remote(global int *src1, global int *src2)  \
{ \
  const int id = get_global_id(0); \
  src2[id] = src1[id]; \
} \
";

void simple_interop()
{
    size_t elemCount = 512;
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

int ocl_l0_interop_p2p(size_t elemCount)
{
    //size_t elemCount = 1024 * 1;
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
    // lzctx0.runKernel("../../lz_p2p/test_kernel_dg2.spv", "local_read_from_remote", lzptr1, lzptr0, elemCount);

    // run p2p data transfer kernel: GPU0 write data to GPU1
    lzctx0.runKernel("../../lz_p2p/test_kernel_dg2.spv", "local_write_to_remote", lzptr1, lzptr0, elemCount);

    // print the content of the original opencl buffers, the data was changed after above level-zero kernel execution.
    oclctx0.printBuffer(clbuf0);
    oclctx1.printBuffer(clbuf1);

    oclctx1.compareBuffer(clbuf1, elemCount, 0 , initBuf);

    oclctx0.freeBuffer(clbuf0);
    oclctx1.freeBuffer(clbuf1);

    return 0;
}

int ocl_p2p()
{
    size_t bytes_size = 15728640;//1024*1024*5;
    size_t elemCount = bytes_size/4;
    std::vector<uint32_t> initBuf(elemCount, 0);
    for (size_t i = 0; i < elemCount; i++)
        initBuf[i] = (i % 1024);

    // initialize two opencl contexts, oclctx0 on GPU0 and oclctx1 on GPU1
    oclContext oclctx0, oclctx1;
    oclctx0.init(0);
    oclctx1.init(1);

    // create clbuf0 on GPU0 and clbuf1 on GPU1
    cl_mem clbuf0 = oclctx0.createBuffer(elemCount * sizeof(uint32_t), initBuf);
    cl_mem clbuf1 = oclctx1.createBuffer(elemCount * sizeof(uint32_t), initBuf);
    oclctx0.printBuffer(clbuf0);
    oclctx1.printBuffer(clbuf1);

    cl_mem clbuf2;
    uint64_t handle1;

    for (int i = 0; i < 4; i++)
    {
        std::cout << std::endl << std::endl << std::endl;
        // derive the handle from clbuf1
        handle1 = oclctx1.deriveHandle(clbuf1);

        // create clbuf2 from handle1
        clbuf2 = oclctx0.createFromHandle(handle1, elemCount * sizeof(uint32_t));

        // use oclctx0 to launch a kernel on GPU0 to write data to remote buffer clbuf2 on GPU1
        oclctx0.runKernel(write_kernel_code, "write_to_remote", clbuf0, clbuf2, elemCount);
        oclctx0.printBuffer(clbuf2);

        //uint64_t handle2 = oclctx1.deriveHandle(clbuf2);

        // use oclctx1 to read the content of the original clbuf1
        //oclctx1.printBuffer(clbuf1);
        // oclctx0.freeBuffer(clbuf2); // cannot released

        //oclctx0.getUsedMem();
        //oclctx1.getUsedMem();
    }
}

int ocl_p2p_copy()
{
    size_t bytes_size = 15728640; // 1024*1024*5;
    size_t elemCount = bytes_size / 4;
    std::vector<uint32_t> initBuf(elemCount, 0);
    for (size_t i = 0; i < elemCount; i++)
        initBuf[i] = ((i * 3) % 1024);

    // initialize two opencl contexts, oclctx0 on GPU0 and oclctx1 on GPU1
    oclContext oclctx0, oclctx1;
    oclctx0.init(0);
    oclctx1.init(1);

    // create clbuf0 on GPU0 and clbuf1 on GPU1
    cl_mem clbuf0 = oclctx0.createBuffer(elemCount * sizeof(uint32_t), initBuf);

    for (size_t i = 0; i < elemCount; i++)
        initBuf[i] = 0;
    cl_mem clbuf1 = oclctx1.createBuffer(elemCount * sizeof(uint32_t), initBuf);
    oclctx0.printBuffer(clbuf0);
    oclctx1.printBuffer(clbuf1);

    cl_mem clbuf2;
    uint64_t handle1;

    for (int i = 0; i < 4; i++)
    {
        std::cout << std::endl
                  << std::endl
                  << std::endl;
        // derive the handle from clbuf1
        handle1 = oclctx1.deriveHandle(clbuf1);

        // create clbuf2 from handle1
        clbuf2 = oclctx0.createFromHandle(handle1, elemCount * sizeof(uint32_t));

        // use oclctx0 to launch a kernel on GPU0 to write data to remote buffer clbuf2 on GPU1
        // remote write
        std::cout << "remote write.........................."<<std::endl;
        oclctx0.copy_data(clbuf0, clbuf2, elemCount);

        // remote read
        std::cout << "remote read.........................."<<std::endl;
        oclctx0.copy_data(clbuf2, clbuf0, elemCount);
        oclctx0.printBuffer(clbuf2);
    }

    oclctx0.freeBuffer(clbuf0);
    oclctx1.freeBuffer(clbuf1);

    return 0;
}

int ocl_p2p_thread_parallel()
{
    size_t bytes_size = 15728640;//1024*1024*5;
    size_t elemCount = bytes_size/4;
    std::vector<uint32_t> initBuf(elemCount, 0);
    for (size_t i = 0; i < elemCount; i++)
        initBuf[i] = (i % 1024);

    // initialize two opencl contexts, oclctx0 on GPU0 and oclctx1 on GPU1
    oclContext oclctx0, oclctx1;
    oclctx0.init(0);
    oclctx1.init(1);

    // create clbuf0 on GPU0 and clbuf1 on GPU1
    cl_mem clbuf00 = oclctx0.createBuffer(elemCount * sizeof(uint32_t), initBuf);
    cl_mem clbuf01 = oclctx0.createBuffer(elemCount * sizeof(uint32_t), initBuf);
    cl_mem clbuf10 = oclctx1.createBuffer(elemCount * sizeof(uint32_t), initBuf);
    cl_mem clbuf11 = oclctx1.createBuffer(elemCount * sizeof(uint32_t), initBuf);
    oclctx0.printBuffer(clbuf00);
    oclctx0.printBuffer(clbuf01);
    oclctx1.printBuffer(clbuf10);
    oclctx1.printBuffer(clbuf11);

    auto p2p_worker = [&](oclContext *local_ctx, cl_mem local_buf, oclContext *remote_ctx, cl_mem remote_buf, size_t elemCount)
    {
        cl_mem clbuf2;
        for (int i = 0; i < 2; i++)
        {
            std::cout << std::endl
                      << std::endl
                      << std::endl;
            // derive the handle from clbuf1
            uint64_t handle = remote_ctx->deriveHandle(remote_buf);

            // create clbuf2 from handle1
            clbuf2 = local_ctx->createFromHandle(handle, elemCount * sizeof(uint32_t));

            // use oclctx0 to launch a kernel on GPU0 to write data to remote buffer clbuf2 on GPU1
            local_ctx->runKernel(write_kernel_code, "write_to_remote", local_buf, clbuf2, elemCount);
            local_ctx->printBuffer(clbuf2);
        }
    };

    if (1)
    {
        std::thread t0(std::bind(p2p_worker, &oclctx0, clbuf00, &oclctx1, clbuf10, elemCount));
        std::thread t1(std::bind(p2p_worker, &oclctx1, clbuf11, &oclctx0, clbuf01, elemCount));

        // std::this_thread::sleep_for(std::chrono::seconds(3));
        t0.join();
        t1.join();
    }
    else
    {
        p2p_worker(&oclctx0, clbuf00, &oclctx1, clbuf10, elemCount);
        p2p_worker(&oclctx1, clbuf11, &oclctx0, clbuf01, elemCount);
    }

    std::cout << "--------------" << std::endl;
    oclctx0.freeBuffer(clbuf00);
    oclctx1.freeBuffer(clbuf10);
    oclctx0.freeBuffer(clbuf01);
    oclctx1.freeBuffer(clbuf11);

    return 0;
}

int main(int argc, char **argv)
{
    //simple_interop();

    if(argc!=2) {
        printf("command:interop <bytes_size_in_KB>\n");
        return 0;
    }
    int bytes=atoi(argv[1]) * 1024;
    ocl_l0_interop_p2p(bytes/4);

    //ocl_p2p();
    //ocl_p2p_copy();
    //ocl_p2p_thread_parallel();

    return 0;
}