//
// Created by Stoorx on 27.04.2021.
//

#include "Application2.h"
#include <iostream>
#include <utility>
#include <CL/cl2.hpp>
#include <random>
#include <fstream>
#include "Application.h"

std::pair<std::vector<int>, double> transpose(
        const std::vector<int>& matrix,
        int m,
        int n
) {
    std::vector<int> r(n * m);
    
    auto begin = std::chrono::system_clock::now();
    
    for(int row = 0; row < n; row++) {
        for(int col = 0; col < m; col++) {
            r[col * n + row] = matrix[row * m + col];
        }
    }
    
    auto end = std::chrono::system_clock::now();
    return {r, (double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000};
}

std::vector<int> initMatrix(int m, int n) {
    std::vector<int> matrix = std::vector<int>(m * n);
    std::mt19937     rnd((std::random_device())());
    
    for(auto& e : matrix) {
        e = (int)(rnd() - (std::numeric_limits<unsigned int>::max() >> 1u));
    }
    return matrix;
}

double getKernelEt(const cl::Event& event) {
    return double(event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                  event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1E6;
}

void printStats(const char* kernelName, double et, int m, int n) {
    std::cout << kernelName << " " << m << " " << n << " " << et << std::endl;
}

void runKernel(
        const cl::Context& context,
        const cl::Program& program,
        const cl::Device& device,
        const char* kernelName,
        int m,
        int n
) {
    cl::CommandQueue commandQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
    std::vector<int> matrix1      = initMatrix(m, n);
    std::vector<int> matrix2      = std::vector<int>(n * m);
    
    cl::Buffer buffer1 = cl::Buffer(context, CL_MEM_READ_ONLY, m * n * sizeof(int));
    cl::Buffer buffer2 = cl::Buffer(context, CL_MEM_WRITE_ONLY, n * m * sizeof(int));
    commandQueue.enqueueWriteBuffer(buffer1, CL_TRUE, 0, m * n * sizeof(int), matrix1.data());
    cl::Kernel kernel = cl::Kernel(program, kernelName);
    
    kernel.setArg(0, buffer1);
    kernel.setArg(1, buffer2);
    kernel.setArg(2, m);
    kernel.setArg(3, n);
    
    cl::Event event;
    
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n, m), cl::NullRange, nullptr, &event);
    commandQueue.enqueueReadBuffer(buffer2, CL_TRUE, 0, sizeof(int) * n * m, matrix2.data());
    commandQueue.finish();
    printStats(kernelName, getKernelEt(event), m, n);
//    Gpu::Application::printMatrix(matrix1, m, n);
//    std::cout << "\n\n";
//    Gpu::Application::printMatrix(matrix2, n, m);
}

void Gpu::Application2::main(const std::vector<std::string>& args) {
    freopen("s.txt", "w", stdout);
    const auto& kernelFilePath = args.at(0);
    const auto m     = std::stoi(args.at(1));
    const auto n     = std::stoi(args.at(2));
    const auto check = args.at(3) == "check";
    
    cl::Platform            platform = cl::Platform::getDefault();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    for(auto& dev : devices) {
        std::cout << "Name: " << dev.getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << "Clock: " << dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
        std::cout << "Compute units: " << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
        std::cout << "Workgroups: " << dev.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
        std::cout << "Local memory: " << dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
        
        std::cout << std::endl;
    }
    cl::Device currentDevice;
    if(devices.size() == 1u) {
        currentDevice = devices.at(0);
    }
    else {
        int id;
        std::cout << "Type number of device >";
        std::cin >> id;
        currentDevice = devices.at(id);
    }
    
    cl::Context   context      = cl::Context(currentDevice);
    std::ifstream source(kernelFilePath);
    cl::string    sourceString = cl::string(std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>());
    cl::Program   program      = cl::Program(context, sourceString);
    program.build(currentDevice);
    
    std::cout << "kernel m n ET\n";
    
    for(int i = 16; i < 10000; i += 16) {
        runKernel(context, program, currentDevice, "transposeSimple", i, i);
        runKernel(context, program, currentDevice, "transposeLocal", i, i);
        runKernel(context, program, currentDevice, "transposeLocalBanksafe", i, i);
    }
    
}
