/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <type_traits>
#include <vector>

#include "raf/ir.h"

#include "ratex/csrc/compiler/raf_lowering_context.h"
#include "ratex/csrc/serialization/serialization.h"

namespace torch_lazy_tensors {
namespace serialization {

using namespace tvm;
using namespace raf::ir;

template <typename T, typename S>
T ToTVMFromLTC(const S& source);

template <typename T, typename S>
T ToLTCFromTVM(const S& source);

// LTC To TVM
template <typename T, typename S>
struct ToTVMFromLTCHelper {
  T operator()(const S& source) const {
    return source;
  }
};

template <>
struct ToTVMFromLTCHelper<Integer, PrimitiveType> {
  Integer operator()(const PrimitiveType& source) const {
    return static_cast<int>(source);
  }
};

template <>
struct ToTVMFromLTCHelper<ObjectRef, LTCShape> {
  ObjectRef operator()(const LTCShape& source) const {
    return Shape(source);
  }
};

// unordered_map -> Map
template <typename TK, typename TV, typename SK, typename SV>
struct ToTVMFromLTCHelper<Map<TK, TV>, std::unordered_map<SK, SV>> {
  Map<TK, TV> operator()(const std::unordered_map<SK, SV>& source) {
    Map<TK, TV> ret;
    for (const auto& kv : source) {
      ret.Set(ToTVMFromLTC<TK>(kv.first), ToTVMFromLTC<TV>(kv.second));
    }
    return ret;
  }
};

// unordered_set -> Array
template <typename T, typename S>
struct ToTVMFromLTCHelper<Array<T>,
                          std::unordered_set<S, tvm::ObjectPtrHash, tvm::ObjectPtrEqual>> {
  Array<T> operator()(
      const std::unordered_set<S, tvm::ObjectPtrHash, tvm::ObjectPtrEqual>& source) {
    Array<T> ret;
    for (const auto& x : source) {
      ret.push_back(ToTVMFromLTC<T>(x));
    }
    return ret;
  }
};

// vector -> Array
template <typename T, typename S>
struct ToTVMFromLTCHelper<Array<T>, std::vector<S>> {
  Array<T> operator()(const std::vector<S>& source) {
    Array<T> ret;
    for (const auto& x : source) {
      ret.push_back(ToTVMFromLTC<T>(x));
    }
    return ret;
  }
};

// span -> Array
template <typename T, typename S>
struct ToTVMFromLTCHelper<Array<T>, lazy_tensors::Span<const S>> {
  Array<T> operator()(const lazy_tensors::Span<const S>& source) {
    Array<T> ret;
    for (const auto& x : source) {
      ret.push_back(ToTVMFromLTC<T>(x));
    }
    return ret;
  }
};

// From TVM To LTC
template <typename T, typename S>
struct ToLTCFromTVMHelper {
  T operator()(const S& source) {
    return source;
  }
};

template <typename T>
struct ToLTCFromTVMHelper<T, Integer> {
  T operator()(const Integer& source) {
    return source.IntValue();
  }
};

template <>
struct ToLTCFromTVMHelper<LTCShape, ObjectRef> {
  LTCShape operator()(const ObjectRef& source) {
    return Downcast<Shape>(source);
  }
};

template <>
struct ToLTCFromTVMHelper<PrimitiveType, Integer> {
  PrimitiveType operator()(const Integer& source) {
    return static_cast<PrimitiveType>(source->value);
  }
};

// Map -> unordered_map
template <typename TK, typename TV, typename SK, typename SV>
struct ToLTCFromTVMHelper<std::unordered_map<TK, TV>, Map<SK, SV>> {
  std::unordered_map<TK, TV> operator()(const Map<SK, SV>& source) {
    std::unordered_map<TK, TV> ret;
    for (const auto& kv : source) {
      ret[ToLTCFromTVM<TK>(kv.first)] = ToLTCFromTVM<TV>(kv.second);
    }
    return ret;
  }
};

// Array -> unordered_set
template <typename T, typename S>
struct ToLTCFromTVMHelper<std::unordered_set<T, tvm::ObjectPtrHash, tvm::ObjectPtrEqual>,
                          Array<S>> {
  std::unordered_set<T, tvm::ObjectPtrHash, tvm::ObjectPtrEqual> operator()(
      const Array<S>& source) {
    std::unordered_set<T, tvm::ObjectPtrHash, tvm::ObjectPtrEqual> ret;
    for (const auto& x : source) {
      ret.insert(ToLTCFromTVM<T>(x));
    }
    return ret;
  }
};

// Array -> vector
template <typename T, typename S>
struct ToLTCFromTVMHelper<std::vector<T>, Array<S>> {
  std::vector<T> operator()(const Array<S>& source) {
    std::vector<T> ret;
    for (const auto& x : source) {
      ret.push_back(ToLTCFromTVM<T>(x));
    }
    return ret;
  }
};

template <typename T, typename S>
T ToTVMFromLTC(const S& source) {
  return ToTVMFromLTCHelper<T, S>()(source);
}

template <typename T, typename S>
T ToLTCFromTVM(const S& source) {
  return ToLTCFromTVMHelper<T, S>()(source);
}

}  // namespace serialization
}  // namespace torch_lazy_tensors
