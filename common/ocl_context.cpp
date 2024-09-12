
#include "ocl_context.h"
#include <chrono>
#include <string>
#include <iostream>

oclContext::oclContext(/* args */)
{
}

oclContext::~oclContext()
{
    printf("Enter %s\n", __FUNCTION__);

    clReleaseCommandQueue(queue_);
    clReleaseContext(context_);
}

void oclContext::init(int devIdx)
{
    cl_int err;
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    CHECK_OCL_ERROR_EXIT(err, "clGetPlatformIDs");

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    CHECK_OCL_ERROR_EXIT(err, "clGetPlatformIDs");

    // char extensions[1024];
    for (const auto &platform : platforms)
    {
        cl_uint num_devices = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);

        // clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, 1024, extensions, NULL);
        // printf("Supported platform extensions: %s\n", extensions);

        if (num_devices > 0)
        {
            platform_ = platform;
            printf("Platform %p has %d GPU devices\n", platform_, num_devices);

            std::vector<cl_device_id> devices(num_devices);
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
            CHECK_OCL_ERROR_EXIT(err, "clGetDeviceIDs");

            if (devIdx >= (num_devices))
            {
                printf("ERROR: don't have OpenCL GPU device for devIdx = %d!\n", devIdx);
                exit(-1);
            }

            device_ = devices[devIdx];
            // clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, 1024, extensions, NULL);
            // printf("Device extensions: %s\n", extensions);

            context_ = clCreateContext(NULL, 1, &device_, NULL, NULL, &err);
            CHECK_OCL_ERROR_EXIT(err, "clCreateContext");

            queue_ = clCreateCommandQueue(context_, device_, 0, &err);
            CHECK_OCL_ERROR_EXIT(err, "clCreateCommandQueue");

            char device_name[1024];
            err = clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
            CHECK_OCL_ERROR_EXIT(err, "clGetDeviceInfo");

            printf("Created device for devIdx = %d on %s, device = %p, contex = %p, queue = %p\n", devIdx, device_name, device_, context_, queue_);

            return;
        }
    }

    printf("ERROR: cannot find OpenCL GPU device!\n");
    exit(-1);
}

void *oclContext::initUSM(size_t elem_count, int offset)
{
    cl_int err;
    void *ptr = nullptr;

    std::vector<uint32_t> hostBuf(elem_count, 0);
    for (size_t i = 0; i < elem_count; i++)
        hostBuf[i] = offset + (i % 1024);

    size_t size = elem_count * sizeof(uint32_t);
    cl_uint alignment = 16;
    ptr = clDeviceMemAllocINTEL(context_, device_, nullptr, size, alignment, &err);
    CHECK_OCL_ERROR_EXIT(err, "clDeviceMemAllocINTEL failed")

    err = clEnqueueMemcpyINTEL(queue_, true, ptr, (void *)hostBuf.data(), size, 0, nullptr, nullptr);
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueMemcpyINTEL failed");

    clFinish(queue_);

    return ptr;
}

void oclContext::readUSM(void *ptr, std::vector<uint32_t> &outBuf, size_t size)
{
    cl_int err;
    err = clEnqueueMemcpyINTEL(queue_, true, (void *)outBuf.data(), ptr, size, 0, nullptr, nullptr);
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueMemcpyINTEL failed");
    clFinish(queue_);
}

void oclContext::freeUSM(void *ptr)
{
    cl_int err;
    err = clMemBlockingFreeINTEL(context_, ptr);
    CHECK_OCL_ERROR(err, "clMemBlockingFreeINTEL");
}

void oclContext::getUsedMem()
{
    size_t used_mem_size;
    clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t),
                    &used_mem_size, NULL);
    printf("gpu mem used: %ld MB\n", used_mem_size / 1024 / 1024);
}

void oclContext::runKernel(char *kernelCode, char *kernelName, void *ptr0, void *ptr1, size_t elemCount)
{
    cl_int err;

    cl_uint knlcount = 1;
    const char *knlstrList[] = {kernelCode};
    size_t knlsizeList[] = {strlen(kernelCode)};

    cl_program program = clCreateProgramWithSource(context_, knlcount, knlstrList, knlsizeList, &err);
    CHECK_OCL_ERROR_EXIT(err, "clCreateProgramWithSource failed");

    std::string buildopt = "-cl-std=CL2.0";
    err = clBuildProgram(program, 0, NULL, buildopt.c_str(), NULL, NULL);
    if (err < 0)
    {
        size_t logsize = 0;
        err = clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
        CHECK_OCL_ERROR_EXIT(err, "clGetProgramBuildInfo failed");

        std::vector<char> logbuf(logsize + 1, 0);
        err = clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, logsize + 1, logbuf.data(), NULL);
        CHECK_OCL_ERROR_EXIT(err, "clGetProgramBuildInfo failed");
        printf("%s\n", logbuf.data());

        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    CHECK_OCL_ERROR_EXIT(err, "clCreateKernel failed");

    err = clSetKernelArgMemPointerINTEL(kernel, 0, ptr0);
    CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg failed");

    err = clSetKernelArgMemPointerINTEL(kernel, 1, ptr1);
    CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg failed");

    size_t global_size[] = {elemCount};
    err = clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr, global_size, nullptr, 0, nullptr, nullptr);
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueNDRangeKernel failed");
    clFinish(queue_);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
}

void oclContext::copy_data(cl_mem buf0, cl_mem buf1, size_t size)
{
    const auto start_1 = std::chrono::high_resolution_clock::now();
    clEnqueueCopyBuffer(queue_, buf0, buf1, 0, 0, size, 0, NULL, NULL);
    clFinish(queue_);
    const auto end_1 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_1 = end_1 - start_1;
    std::cout << "copy host time: " << elapsed_1.count() << " ms, size = " << size / 1024 << "KB, bandwidth = " << size * 1000 / 1024 / 1024 / 1024 / elapsed_1.count() << " GB / s " << std::endl
              << std::endl;
}

void oclContext::runKernel(char *kernelCode, char *kernelName, cl_mem buf0, cl_mem buf1, size_t elemCount)
{
    cl_int err;

    const auto start_1 = std::chrono::high_resolution_clock::now();
    cl_uint knlcount = 1;
    const char *knlstrList[] = {kernelCode};
    size_t knlsizeList[] = {strlen(kernelCode)};

    if (inited_ == false)
    {
        inited_ = true;
        program = clCreateProgramWithSource(context_, knlcount, knlstrList, knlsizeList, &err);
        CHECK_OCL_ERROR_EXIT(err, "clCreateProgramWithSource failed");

        std::string buildopt = "-cl-std=CL2.0 -cl-intel-greater-than-4GB-buffer-required";
        err = clBuildProgram(program, 0, NULL, buildopt.c_str(), NULL, NULL);
        if (err < 0)
        {
            size_t logsize = 0;
            err = clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsize);
            CHECK_OCL_ERROR_EXIT(err, "clGetProgramBuildInfo failed");

            std::vector<char> logbuf(logsize + 1, 0);
            err = clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, logsize + 1, logbuf.data(), NULL);
            CHECK_OCL_ERROR_EXIT(err, "clGetProgramBuildInfo failed");
            printf("%s\n", logbuf.data());

            exit(1);
        }

        kernel = clCreateKernel(program, kernelName, &err);
        CHECK_OCL_ERROR_EXIT(err, "clCreateKernel failed");
    }

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf0);
    CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg failed");

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf1);
    CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg failed");

    size_t global_size[] = {elemCount};
    err = clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr, global_size, nullptr, 0, nullptr, nullptr);
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueNDRangeKernel failed");
    clFinish(queue_);

    const auto end_1 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_1 = end_1 - start_1;
    std::cout << "ocl host time: " << elapsed_1.count() << " ms, size = " << elemCount * 4 / 1024 << "KB, bandwidth = " << elemCount * 4.0 * 1000 / 1024 / 1024 / 1024 / elapsed_1.count() << " GB / s " << std::endl
              << std::endl;

    if (inited_ == false)
    {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
    }
}

cl_mem oclContext::createBuffer(size_t size, const std::vector<uint32_t> &inbuf)
{
    cl_int err;

    cl_mem clbuf = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_ALLOW_UNRESTRICTED_SIZE_INTEL, size, nullptr, &err);
    CHECK_OCL_ERROR_EXIT(err, "clCreateBuffer");

    if (!inbuf.empty())
    {
        err = clEnqueueWriteBuffer(queue_, clbuf, CL_TRUE, 0, size, inbuf.data(), 0, NULL, NULL);
        CHECK_OCL_ERROR_EXIT(err, "clEnqueueWriteBuffer failed");

        clFinish(queue_);
    }

    return clbuf;
}

uint64_t oclContext::deriveHandle(cl_mem clbuf)
{
    cl_int err;
    uint64_t nativeHandle;
    err = clGetMemObjectInfo(clbuf, CL_MEM_ALLOCATION_HANDLE_INTEL, sizeof(nativeHandle), &nativeHandle, NULL);

    CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_ALLOCATION_HANDLE_INTEL failed");
    printf("deriveHandle: cl_mem = %p, fd = %ld\n", clbuf, nativeHandle);

    return nativeHandle;
}

uint64_t oclContext::deriveHandle(void *usm_buf)
{
    cl_int err;
    uint64_t nativeHandle;
    err = clGetMemAllocInfoINTEL(context_, usm_buf, CL_MEM_ALLOCATION_HANDLE_INTEL, sizeof(nativeHandle), &nativeHandle, NULL);

    CHECK_OCL_ERROR(err, "clGetMemAllocInfoINTEL - CL_MEM_ALLOCATION_HANDLE_INTEL failed");

    return nativeHandle;
}

cl_mem oclContext::createFromHandle(uint64_t handle, size_t size)
{
    cl_int err;

    // Create extMemBuffer of type cl_mem from fd.
    cl_mem_properties extMemProperties[] = {
        (cl_mem_properties)CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR,
        (cl_mem_properties)handle,
        0};

    cl_mem extMemBuffer = clCreateBufferWithProperties(context_, extMemProperties, 0, size, NULL, &err);
    CHECK_OCL_ERROR(err, "clCreateBufferWithProperties failed");

    size_t data_size = 0;
    err = clGetMemObjectInfo(extMemBuffer, CL_MEM_SIZE, sizeof(size_t), &data_size, NULL);
    CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_SIZE failed");
    printf("size = %ld, data_size = %ld\n", size, data_size);

    cl_mem_object_type type;
    err = clGetMemObjectInfo(extMemBuffer, CL_MEM_TYPE, sizeof(type), &type, NULL);
    CHECK_OCL_ERROR(err, "clGetMemObjectInfo - CL_MEM_TYPE failed");
    printf("size = %ld, remote_type = %d\n", size, type);

    return extMemBuffer;
}

void oclContext::releaseMemFromHandle(cl_mem extMemBuffer)
{
    // clEnqueueReleaseExternalMemObjectsKHR(queue_,1,&extMemBuffer,0,NULL,NULL);
    clReleaseMemObject(extMemBuffer);
}

void oclContext::readBuffer(cl_mem clbuf, std::vector<uint32_t> &outBuf, size_t size, size_t offset)
{
    cl_int err;
    err = clEnqueueReadBuffer(queue_, clbuf, CL_TRUE, offset, size, outBuf.data(), 0, NULL, NULL);
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueReadBuffer failed");
    clFinish(queue_);
}

void oclContext::freeBuffer(cl_mem clbuf)
{
    clReleaseMemObject(clbuf);
}

void oclContext::printBuffer(cl_mem clbuf, size_t count, size_t offset)
{
    std::vector<uint32_t> outBuf(count, 0);
    readBuffer(clbuf, outBuf, count * sizeof(uint32_t), offset);

    printf("The first %d elements in cl_mem = %p are: \n", count, clbuf);
    for (int i = 0; i < count; i++)
    {
        printf("%d, ", outBuf[i]);
        if (i && i % 16 == 0)
            printf("\n");
    }
    printf("\n");
}

void oclContext::compareBuffer(cl_mem clbuf, size_t count, size_t offset, std::vector<uint32_t> &src)
{
    std::vector<uint32_t> outBuf(count, 0);
    readBuffer(clbuf, outBuf, count * sizeof(uint32_t), offset);

    int total = 0;
    for (int i = 0; i < count; i++)
    {
        if (outBuf[i] != 5 * src[i])
            total++;
    }

    std::cout << std::endl;
    if (total == 0)
    {
        std::cout << "Compare data result: Pass (100\%)" << std::endl
                  << std::endl;
    }
    else
    {
        std::cout << "Compare data result: Failed (" << 100.0 * total / count << "\%)" << std::endl
                  << std::endl;
    }
}