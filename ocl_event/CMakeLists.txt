add_executable(ocl_event main.cpp)
add_executable(ocl_event_2 main_func.cpp)


target_link_libraries(ocl_event OpenCL)
target_link_libraries(ocl_event_2 OpenCL)
