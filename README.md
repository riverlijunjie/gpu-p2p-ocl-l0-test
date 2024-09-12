# level-zero-p2p

## build projects

```bash
sudo apt update
sudo apt install opencl-headers
sudo apt install ocl-icd-opencl-dev

cd gpu-p2p-test
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-O0 -g"
make
```

## run tests

```bash
cd build/lz_p2p
./lzp2p -l 0 -r 1 -n 4m

cd build/ocl_p2p
./oclp2p

cd build/interop
./interop

# run with drm trace
sudo apt update
sudo apt install trace-cmd
cd build/lz_p2p
cp ../../auto.py ./
python ./auto.py "./lzp2p -l 0 -r 1 -n 4m"
```

lz_p2p Results

```
(base) gta@xxxxxx:~/data/lz_samples/level-zero-p2p/lz_p2p/build$ ./p2p -l 0 -r 1 -n 16m
#### Input parameters: loca_ gpu idx = 0, remote_gpu idx = 1, data_count = 16777216
INFO: Enter lzContext 
INFO: Enter lzContext 
INFO: driver count = 1
#### device count = [0/2], devcie_name = Intel(R) Data Center GPU Flex 170
Found 1 device...
Driver version: 17002962
API version: 65539
INFO: find device handle = 0x5654b5a964d0
INFO: driver count = 1
#### device count = [0/2], devcie_name = Intel(R) Data Center GPU Flex 170
#### device count = [1/2], devcie_name = Intel(R) Data Center GPU Flex 170
Found 1 device...
Driver version: 17002962
API version: 65539
INFO: find device handle = 0x5654b5a9e190
queryP2P, dev0 = 0x5654b5a964d0, dev1 = 0x5654b5a9e190, flags = 1
queryP2P, dev0 = 0x5654b5a9e190, dev1 = 0x5654b5a964d0, flags = 1
buf0 = 0xffff80ac00200000, buf1 = 0xffff80d600200000
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
Kernel timestamp statistics (prior to V1.2): 
        Global start : 1157526896 cycles
        Kernel start: 333686 cycles
        Kernel end: 8036576 cycles
        Global end: 1165229784 cycles
        timerResolution: 52 ns
        Kernel duration : 7702890 cycles
        Kernel Time: 400550.280000 us
#### gpuKernelTime = 400550.280000, elemCount = 16777216, Bandwidth = 0.167542 GB/s
3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 
Kernel timestamp statistics (prior to V1.2): 
        Global start : 1165369906 cycles
        Kernel start: 8139744 cycles
        Kernel end: 8303944 cycles
        Global end: 1165534104 cycles
        timerResolution: 52 ns
        Kernel duration : 164200 cycles
        Kernel Time: 8538.400000 us
#### gpuKernelTime = 8538.400000, elemCount = 16777216, Bandwidth = 7.859653 GB/s
15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 
done
INFO: Enter ~lzContext 
INFO: Enter ~lzContext
```

ocl_p2p Results

```
Platform 0x5594724d5970 has 2 GPU devices
Created device for devIdx = 0 on Intel(R) Data Center GPU Flex 170, device = 0x5594724d5a40, contex = 0x5594724dad20, queue = 0x55947154d260
Platform 0x5594724d5970 has 2 GPU devices
Created device for devIdx = 0 on Intel(R) Data Center GPU Flex 170, device = 0x5594724d5a40, contex = 0x55947154eb10, queue = 0x5594724e51e0
buf0 = 0xffff8081fff60000, buf1 = 0xffff8081ffdf0000
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
ERROR: oclContext::runKernel, line = 138, clSetKernelArg failed! err = -50 (CL_INVALID_ARG_VALUE)
```

interop P2P results

```
(base) gta@DUT7113ATSM:~/data/gpu_p2p_code/gpu-p2p-test/build/interop$ ./interop 16 
Platform 0x56ba3ceb60e0 has 2 GPU devices
Created device for devIdx = 0 on Intel(R) Arc(TM) A770 Graphics, device = 0x56ba3ceb61b0, contex = 0x56ba3cebd9b0, queue = 0x56ba3bd21ee0
Platform 0x56ba3ceb60e0 has 2 GPU devices
Created device for devIdx = 1 on Intel(R) Arc(TM) A770 Graphics, device = 0x56ba3ceb8120, contex = 0x56ba3bd236f0, queue = 0x56ba3bd26b90
INFO: Enter lzContext 
INFO: Enter lzContext 
INFO: driver count = 1
#### device count = [0/2], devcie_name = Intel(R) Arc(TM) A770 Graphics
Found 1 device...
Driver version: 17002026
API version: 65539
INFO: find device handle = 0x56ba3e42d3e0
INFO: driver count = 1
#### device count = [0/2], devcie_name = Intel(R) Arc(TM) A770 Graphics
#### device count = [1/2], devcie_name = Intel(R) Arc(TM) A770 Graphics
Found 1 device...
Driver version: 17002026
API version: 65539
INFO: find device handle = 0x56ba3e435380
The first 16 elements in cl_mem = 0x56ba3e45ce90 are: 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
The first 16 elements in cl_mem = 0x56ba3e460710 are: 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
deriveHandle: cl_mem = 0x56ba3e45ce90, fd = 9
deriveHandle: cl_mem = 0x56ba3e460710, fd = 10
handle = 9, bufSize = 16384
MemAllocINFO: memory = 0xffff81d5ff800000, stype = 0, pNext = 0x00000000, type = 2, id = 0x00000002, pagesize = 0
handle = 10, bufSize = 16384
MemAllocINFO: memory = 0xffff81ffff600000, stype = 0, pNext = 0x00000000, type = 2, id = 0x00000003, pagesize = 0
The first 16 elements in level-zero ptr = 0xffff81d5ff800000 are: 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
The first 16 elements in level-zero ptr = 0xffff81ffff600000 are: 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
Kernel timestamp statistics (prior to V1.2): 
        Global start : 2380304566 cycles
        Kernel start: 220870 cycles
        Kernel end: 221496 cycles
        Global end: 2380305194 cycles
        timerResolution: 52 ns
        Kernel duration : 626 cycles
        Kernel Time: 32.552000 us
#### gpuKernelTime = 32.552000, elemCount = 4096, Bandwidth = 0.503318 GB/s
The first 16 elements in cl_mem = 0x56ba3e45ce90 are: 
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
The first 16 elements in cl_mem = 0x56ba3e460710 are: 
0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 

Compare data result: Pass (100%)

INFO: Enter ~lzContext 
INFO: Enter ~lzContext 
Enter ~oclContext
Enter ~oclContext
```
