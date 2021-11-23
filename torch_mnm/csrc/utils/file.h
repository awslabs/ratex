#pragma once

#include <fstream>
#include <iostream>
#include <sys/stat.h>

namespace torch_mnm {

bool PathExist(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

std::string Load(std::ifstream& in) {
    std::ostringstream sstr;
    sstr << in.rdbuf();
    return sstr.str();
}

void Save(std::ofstream& out, std::string str) {
    out << str;
    out.flush();
}

}  // namespace torch_mnm
