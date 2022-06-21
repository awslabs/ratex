/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/lowering_context.h"

#include "raf/device.h"
#include "raf/ir.h"
#include "raf/value.h"
#include "raf/pass.h"
#include "raf/binding.h"
#include "raf/op_utils.h"

namespace raf {
namespace pass {
namespace extract_binding {

ir::Expr ExtractBinding(const ir::Var& var, const ir::Array<ir::Var>& ignore);

}  // namespace extract_binding
}  // namespace pass
}  // namespace raf

namespace raf {

using namespace lazy_tensors;

template <>
inline DType::operator PrimitiveType() const {
  switch (code) {
    case kDLInt:
      if (bits == 8) return PrimitiveType::S8;
      if (bits == 16) return PrimitiveType::S16;
      if (bits == 32) return PrimitiveType::S32;
      if (bits == 64) return PrimitiveType::S64;
      break;
    case kDLUInt:
      if (bits == 1) return PrimitiveType::PRED;
      if (bits == 8) return PrimitiveType::U8;
      break;
    case kDLFloat:
      if (bits == 16) return PrimitiveType::F16;
      if (bits == 32) return PrimitiveType::F32;
      if (bits == 64) return PrimitiveType::F64;
    case kDLBfloat:
      if (bits == 16) return PrimitiveType::BF16;
    default:
      LTC_LOG(FATAL) << "Not implemented yet: " << c_str();
  }
}

}  // namespace raf

namespace torch_lazy_tensors {
namespace compiler {
namespace raf_backend {

using namespace lazy_tensors;
using namespace raf;
using namespace raf::value;
using namespace raf::ir;
using namespace raf::tensor;

inline Shape WithDefaultMinorToMajor(const Shape& shape) {
  Shape ret = shape;
  Layout* layout = ret.mutable_layout();
  for (int i = shape.dimensions_size() - 1; i >= 0; i--) {
    layout->add_minor_to_major(i);
  }
  return ret;
}

inline Shape ToLTCShape(std::vector<int64_t> shape, raf::DType dtype) {
  return WithDefaultMinorToMajor(Shape(dtype, lazy_tensors::Span<const int64_t>(shape)));
}

inline Shape ToLTCShape(const Type& type) {
  if (const auto* ttype = type.as<TensorTypeNode>()) {
    std::vector<int64_t> shape;
    for (auto dim : ttype->shape) {
      auto dim_imm = dim.as<IntImmNode>();
      CHECK(dim_imm);
      shape.push_back(dim_imm->value);
    }
    return ToLTCShape(shape, ttype->dtype.operator DLDataType());
  } else if (const auto* ttype = type.as<TupleTypeNode>()) {
    std::vector<Shape> fields;
    for (const auto& ty : ttype->fields) {
      fields.push_back(ToLTCShape(ty));
    }
    return Shape(fields);
  } else if (const auto* ftype = type.as<FuncTypeNode>()) {
    // FuncType cannot be expressed in Shape. We simply put its ret_type in Shape
    return ToLTCShape(ftype->ret_type);
  }
  LTC_LOG(FATAL) << "NotImplementedError: " << type;
}

inline DType ToRAFDType(const PrimitiveType type) {
  switch (type) {
    case PrimitiveType::S8:
      return DType(DTypeCode::kInt(), 8);
    case PrimitiveType::S64:
      return DType(DTypeCode::kInt(), 64);
    case PrimitiveType::PRED:
      return DType(DTypeCode::kUInt(), 1);
    case PrimitiveType::U8:
      return DType(DTypeCode::kUInt(), 8);
    case PrimitiveType::F16:
      return DType(DTypeCode::kFloat(), 16);
    case PrimitiveType::F32:
      return DType(DTypeCode::kFloat(), 32);
    case PrimitiveType::F64:
      return DType(DTypeCode::kFloat(), 64);
    case PrimitiveType::BF16:
      return DType(DTypeCode::kBFloat(), 16);
    default:
      LTC_LOG(FATAL) << type << " Not implemented yet.";
  }
}

inline std::tuple<std::vector<int64_t>, DType> ToRAFShape(
    const lazy_tensors::client::ShapeData& shape) {
  return std::make_tuple(shape.dimensions(), ToRAFDType(shape.element_type()));
}

inline std::tuple<std::vector<int64_t>, DType> ToRAFShape(const Shape& shape) {
  lazy_tensors::Span<const int64_t> dimension = shape.dimensions();
  return std::make_tuple(std::vector<int64_t>(dimension.begin(), dimension.end()),
                         ToRAFDType(shape.element_type()));
}

inline Type ToRAFType(const Shape& shape) {
  std::vector<int64_t> vec_shape;
  DType dtype;
  Array<tvm::PrimExpr> arr_shape;
  std::tie(vec_shape, dtype) = ToRAFShape(shape);
  for (const auto& x : vec_shape) {
    arr_shape.push_back(Integer(x));
  }
  return TensorType(arr_shape, DataType(dtype.operator DLDataType()));
}

inline raf::Device ToRAFDevice(const std::string& device) {
  LTC_CHECK(device != "") << "device is empty";
  auto sep = device.find(":");
  std::string dev_type = device.substr(0, sep);
  std::transform(dev_type.begin(), dev_type.end(), dev_type.begin(), ::tolower);
  int dev_id = (sep == std::string::npos) ? 0 : std::stoi(device.substr(sep + 1));

  if (dev_type == "cpu") {
    return raf::Device(DevType::kCPU(), dev_id);
  } else if (dev_type == "gpu" || dev_type == "cuda") {
    return raf::Device(DevType::kCUDA(), dev_id);
  }
  LTC_LOG(FATAL) << "Not supported device: " << device;
}

inline std::tuple<Var, Var> PromoteDType(const Var& op0, const Var& op1) {
  using namespace raf::binding;
  auto tty0 = Downcast<TensorType>(op0->checked_type());
  auto tty1 = Downcast<TensorType>(op1->checked_type());
  LTC_CHECK_EQ(tty0->dtype.lanes(), tty1->dtype.lanes());
  if (tty0->dtype.code() != tty1->dtype.code()) {
    if (tty0->dtype.is_float() && tty1->dtype.is_int()) {
      return std::make_tuple(op0,
                             BindSymbol(raf::ir::Call(Op::Get("raf.op.cast_like"), {op1, op0})));
    } else if (tty1->dtype.is_float() && tty0->dtype.is_int()) {
      return std::make_tuple(BindSymbol(raf::ir::Call(Op::Get("raf.op.cast_like"), {op0, op1})),
                             op1);
    }
    LTC_LOG(FATAL) << "Not implemented yet.";
  }
  if (tty0->dtype.bits() == tty1->dtype.bits()) {
    return std::make_tuple(op0, op1);
  } else if (tty0->dtype.bits() < tty1->dtype.bits()) {
    return std::make_tuple(BindSymbol(raf::ir::Call(Op::Get("raf.op.cast_like"), {op0, op1})), op1);
  } else {
    return std::make_tuple(op0, BindSymbol(raf::ir::Call(Op::Get("raf.op.cast_like"), {op1, op0})));
  }
}

// Dispatch DataType to the C++ data type
#define RAF_DTYPE_DISPATCH(type, CDType, ...)                     \
  if (type == raf::DType(DTypeCode::kFloat(), 64)) {              \
    typedef double CDType;                                        \
    { __VA_ARGS__ }                                               \
  } else if (type == raf::DType(DTypeCode::kFloat(), 32)) {       \
    typedef float CDType;                                         \
    { __VA_ARGS__ }                                               \
  } else if (type == raf::DType(DTypeCode::kFloat(), 16)) {       \
    typedef uint16_t CDType;                                      \
    { __VA_ARGS__ }                                               \
  } else if (type == raf::DType(DTypeCode::kBFloat(), 16)) {      \
    typedef uint16_t CDType;                                      \
    { __VA_ARGS__ }                                               \
  } else if (type == raf::DType(DTypeCode::kInt(), 64)) {         \
    typedef int64_t CDType;                                       \
    { __VA_ARGS__ }                                               \
  } else if (type == raf::DType(DTypeCode::kInt(), 32)) {         \
    typedef int32_t CDType;                                       \
    { __VA_ARGS__ }                                               \
  } else if (type == raf::DType(DTypeCode::kInt(), 16)) {         \
    typedef int16_t CDType;                                       \
    { __VA_ARGS__ }                                               \
  } else if (type == raf::DType(DTypeCode::kInt(), 8)) {          \
    typedef int8_t CDType;                                        \
    { __VA_ARGS__ }                                               \
  } else if (type == raf::DType(DTypeCode::kUInt(), 64)) {        \
    typedef uint64_t CDType;                                      \
    { __VA_ARGS__ }                                               \
  } else if (type == raf::DType(DTypeCode::kUInt(), 32)) {        \
    typedef uint32_t CDType;                                      \
    { __VA_ARGS__ }                                               \
  } else if (type == raf::DType(DTypeCode::kUInt(), 16)) {        \
    typedef uint16_t CDType;                                      \
    { __VA_ARGS__ }                                               \
  } else if (type == raf::DType(DTypeCode::kUInt(), 8)) {         \
    typedef uint8_t CDType;                                       \
    { __VA_ARGS__ }                                               \
  } else {                                                        \
    LOG(FATAL) << "unknown data type " << type.code << type.bits; \
  }

// Create a RelayConstant with a scalar
template <typename T>
inline RelayConstant MakeConstantScalar(raf::DType dtype, T value, const raf::Device& to_dev) {
  // FIXME(multi-node): This can be a problem when we use multi-node environment
  tvm::runtime::NDArray cpu_array = tvm::runtime::NDArray::Empty({1}, dtype, {kDLCPU, 0});
  RAF_DTYPE_DISPATCH(dtype, CDType, {
    if (dtype == raf::DType(DTypeCode::kFloat(), 16)) {
      // convert to float16
      // storage is uint16_t
      *static_cast<CDType*>(cpu_array->data) =
          __truncXfYf2__<float, uint32_t, 23, uint16_t, uint16_t, 10>(static_cast<float>(value));
    } else if (dtype == raf::DType(DTypeCode::kBFloat(), 16)) {
      // convert to bfloat16
      // storage is uint16_t
      *static_cast<CDType*>(cpu_array->data) = c10::BFloat16(static_cast<float>(value)).x;
    } else {
      *static_cast<CDType*>(cpu_array->data) = value;
    }
  })
  auto tgt_array = tvm::runtime::NDArray::Empty({1}, dtype, to_dev);
  tgt_array.CopyFrom(cpu_array);
  return MakeConstant(TensorValue::make(Tensor::FromDLPack(tgt_array.ToDLPack())));
}

// Convert the replica groups from vector to tuple value
inline TupleValue ConvertReplicaGroupsToValue(std::vector<std::vector<int64_t>> replica_groups) {
  Array<Value> ret;
  for (auto const& group : replica_groups) {
    ret.push_back(raf::op::ArrayToIntTuple(group));
  }
  return TupleValue::make(ret);
}

}  // namespace raf_backend
}  // namespace compiler
}  // namespace torch_lazy_tensors
