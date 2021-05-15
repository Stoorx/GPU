//
// Created by Stoorx on 22.04.2021.
//

#pragma once

#include <string>
#include <vector>

namespace Gpu {
  class Application {
    public:
      static void main(const std::vector<std::string>& args);
      
      static std::pair<std::vector<int>, double> mul(
              const std::vector<int>& m1,
              const std::vector<int>& m2,
              int m, int k, int n
      );
    
      static void printMatrix(const std::vector<int>& m, int w, int h);
  };
}