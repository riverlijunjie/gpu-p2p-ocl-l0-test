#pragma once

#include "common.h"

class oclContext
{
private:
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    bool inited_ = false;
    cl_program program;
    cl_kernel kernel;

public:
    oclContext(/* args */);
    ~oclContext();

    cl_device_id device() { return device_; };
    cl_context context() { return context_; };
    cl_command_queue queue() { return queue_; };

    void init(int devIdx);
    void *initUSM(size_t elem_count, int offset);
    void readUSM(void *ptr, std::vector<uint32_t> &outBuf, size_t size);
    void freeUSM(void *ptr);
    void runKernel(char *programFile, char *kernelName, void *ptr0, void *ptr1, size_t elemCount);
    void runKernel(char *programFile, char *kernelName, cl_mem buf0, cl_mem buf1, size_t elemCount);

    cl_mem createBuffer(size_t size, const std::vector<uint32_t> &inbuf = std::vector<uint32_t>{});
    uint64_t deriveHandle(cl_mem clbuf);
    uint64_t deriveHandle(void* usm_buf);
    void releaseMemFromHandle(cl_mem extMemBuffer);
    void getUsedMem();
    cl_mem createFromHandle(uint64_t handle, size_t size);
    void readBuffer(cl_mem clbuf, std::vector<uint32_t> &outBuf, size_t size, size_t offset);
    void freeBuffer(cl_mem clbuf);
    void copy_data(cl_mem buf0, cl_mem buf1,size_t elemCount);
    void printBuffer(cl_mem clbuf, size_t count = 16, size_t offset = 0);
    void compareBuffer(cl_mem clbuf, size_t count, size_t offset, std::vector<uint32_t>& src);
};