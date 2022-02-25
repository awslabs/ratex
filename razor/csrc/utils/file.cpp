/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "file.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

namespace razor {

bool PathExist(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

std::string Load(const std::string& file_path) {
  std::ifstream in(file_path);
  std::ostringstream sstr;
  sstr << in.rdbuf();
  return sstr.str();
}

void Save(const std::string& file_path, const std::string& str) {
  std::ofstream out(file_path);
  out << str;
  out.flush();
  out.close();
}

void CopyFile(const std::string& from, const std::string& to) {
  std::ifstream ifs(from, std::ios::binary);
  std::ofstream ofs(to, std::ios::binary);
  ofs << ifs.rdbuf();
  ofs.flush();
  ifs.close();
  ofs.close();
}

}  // namespace razor
