#include <iostream>
#include <CL/opencl.hpp>
#include <CL/cl.h>
#include <vector>
#include <string>
#include "Application.h"
#include "Application2.h"

int main(int argc, char** argv) {
    try {
        std::vector<std::string> args;
        args.reserve(argc - 1);
        for(size_t i = 1; i < argc; i++) {
            args.emplace_back(argv[i]);
        }
        Gpu::Application2::main(args);
    }
    catch(const std::exception& e) {
        std::cerr << "Unhandled exception:\n" << e.what() << std::endl;
        return -1;
    }
    return 0;
}
