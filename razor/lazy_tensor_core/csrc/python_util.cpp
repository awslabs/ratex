/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/python_util.h"

#include <Python.h>
#include <frameobject.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/python_strings.h>

namespace torch_lazy_tensors {

absl::optional<SourceLocation> GetPythonFrameTop() {
  if (!Py_IsInitialized()) {
    return absl::nullopt;
  }
  pybind11::gil_scoped_acquire gil;
  PyFrameObject* frame = PyEval_GetFrame();
  if (frame == nullptr) {
    return absl::nullopt;
  }
  SourceLocation loc;
  loc.line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
  loc.file = THPUtils_unpackString(frame->f_code->co_filename);
  loc.function = THPUtils_unpackString(frame->f_code->co_name);
  return loc;
}

std::vector<SourceLocation> GetPythonFrames() {
  std::vector<SourceLocation> frames;
  if (Py_IsInitialized()) {
    pybind11::gil_scoped_acquire gil;
    PyFrameObject* frame = PyEval_GetFrame();
    while (frame != nullptr) {
      SourceLocation loc;
      loc.line = PyCode_Addr2Line(frame->f_code, frame->f_lasti);
      loc.file = THPUtils_unpackString(frame->f_code->co_filename);
      loc.function = THPUtils_unpackString(frame->f_code->co_name);
      frames.push_back(std::move(loc));
      frame = frame->f_back;
    }
  }
  return frames;
}

std::ostream& operator<<(std::ostream& stream, const std::vector<SourceLocation>& frames) {
  stream << "Python Frames:\n";
  for (auto& location : frames) {
    stream << "  " << location.function << " (" << location.file << ":" << location.line << ")\n";
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const SourceLocation& frame) {
  stream << "  " << frame.function << " (" << frame.file << ":" << frame.line << ")\n";
}

}  // namespace torch_lazy_tensors
