//
// Created by Stoorx on 22.04.2021.
//

#include "Application.h"
#include <iostream>
#include <functional>
#include <utility>
#include <CL/cl2.hpp>
#include <CL/opencl.hpp>
#include <random>
#include <fstream>

std::pair<std::vector<float>, double> mul(const std::vector<float>& m1, const std::vector<
        float>& m2, int m, int k, int n) {
    std::vector<float> r(m * n);
    
    auto begin = std::chrono::system_clock::now();
    
    for(int row = 0; row < m; row++) {
        for(int col = 0; col < n; col++) {
            float sum = 0;
            
            for(int i = 0; i < k; i++) {
                sum += m1[row * k + i] * m2[i * n + col];
            }
            
            r[col * m + row] = sum;
        }
    }
    
    auto end = std::chrono::system_clock::now();
    return {r, (double)std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()};
}

void printMatrix(const std::vector<float>& m, int w, int h) {
    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            std::cout << m[i * w + j] << " ";
        }
        std::cout << std::endl;
    }
}

void Gpu::Application::main(const std::vector<std::string>& args) {
    const auto& kernelFilePath = args.at(0);
    const auto m     = std::stoi(args.at(1));
    const auto n     = std::stoi(args.at(2));
    const auto k     = std::stoi(args.at(3));
    const auto check = args.at(4) == "check";
    
    cl::Platform            platform = cl::Platform::getDefault();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    for(auto& dev : devices) {
        std::cout << dev.getInfo<CL_DEVICE_NAME>() << std::endl;
        std::cout << dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
        std::cout << dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
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
    
    cl::Context      context      = cl::Context(currentDevice);
    cl::CommandQueue commandQueue = cl::CommandQueue(context, currentDevice);
    
    std::vector<float> matrix1 = std::vector<float>(m * k);
    std::vector<float> matrix2 = std::vector<float>(k * n);
    std::vector<float> matrix3 = std::vector<float>(m * n);
    
    std::mt19937 rnd((std::random_device())());
    
    for(auto& e : matrix1) {
        e = (float)(rnd() - (std::numeric_limits<unsigned int>::max() >> 1u)) / 1000.0f;
    }
    for(auto& e : matrix2) {
        e = (float)(rnd() - (std::numeric_limits<unsigned int>::max() >> 1u)) / 1000.0f;
    }
    
    
    cl::Buffer buffer1    = cl::Buffer(context, matrix1.begin(), matrix1.end(), true);
    cl::Buffer buffer2    = cl::Buffer(context, matrix2.begin(), matrix2.end(), true);
    cl::Buffer buffer3    = cl::Buffer(context, CL_MEM_WRITE_ONLY, m * n * sizeof(float));
    
    std::ifstream source(kernelFilePath);
    cl::Program   program = cl::Program(
            context,
            cl::string(
                    std::istreambuf_iterator<char>(source),
                    std::istreambuf_iterator<char>()
            )
    );
    
    program.build(currentDevice);
    
    cl::Kernel kernel = cl::Kernel(program, "mul");
    
    kernel.setArg(0, buffer1);
    kernel.setArg(1, buffer2);
    kernel.setArg(2, buffer3);
    kernel.setArg(3, m);
    kernel.setArg(4, n);
    kernel.setArg(5, k);
    
    cl::Event event;
    
    cl::NDRange global_work_group_size(std::min(currentDevice.getInfo<
            CL_DEVICE_MAX_WORK_GROUP_SIZE>(), (unsigned int)std::min(std::min(m, n), k)));
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_group_size, cl::NullRange, nullptr, &event);
    commandQueue.enqueueReadBuffer(buffer3, CL_TRUE, 0, sizeof(float) * m * n, matrix3.data());
    matrix3 = mul(matrix1, matrix2, m, k, n).first;
    commandQueue.finish();
    
    std::cout << "ElapsedTime: "
              << double(event.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                        event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / 1000 << std::endl;
    
    if(check) {
        auto cpu = mul(matrix1, matrix2, m, k, n);
        std::cout << "CpuElapsedTime: " << cpu.second << std::endl;
        std::cout << (cpu.first == matrix3 ? "Equals" : "Not equals") << std::endl;
        printMatrix(matrix3, m, n);
        std::cout << std::endl;
        printMatrix(cpu.first, m, n);
    }
}
