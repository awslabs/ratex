#pragma once

#include <fstream>
#include <iostream>
#include <sys/stat.h>

namespace torch_mnm {

bool PathExist(const std::string& path);

std::string Load(const std::string& file_path);

void Save(const std::string& dir, const std::string& str);

void CopyFile(const std::string& from, const std::string& to);

}  // namespace torch_mnm
