/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <fstream>
#include <iostream>
#include <sys/stat.h>

namespace ratex {

bool PathExist(const std::string& path);

std::string Load(const std::string& file_path);

void Save(const std::string& dir, const std::string& str);

void CopyFile(const std::string& from, const std::string& to);

std::string GetBasename(std::string file_path);

std::string GetParentPath(const std::string& path);

}  // namespace ratex
