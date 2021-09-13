#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/lowering_context.h"

#include "mnm/base.h"
#include "mnm/ir.h"
#include "mnm/value.h"
#include "mnm/pass.h"
#include "mnm/binding.h"

namespace mnm {
namespace pass {
namespace extract_binding {
  
ir::Expr ExtractBinding(const ir::Var& var, const ir::Array<ir::Var>& ignore);

}  // namespace extract_binding
}  // namespace pass
}  // namespace mnm

namespace mnm {

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
  }
  LTC_LOG(FATAL) << "Not implemented yet: " << c_str();
}

}  // namespace mnm

namespace torch_lazy_tensors {
namespace compiler {
namespace mnm_backend {

using namespace lazy_tensors;
using namespace mnm;
using namespace mnm::value;
using namespace mnm::ir;

inline Shape WithDefaultMinorToMajor(const Shape& shape) {
  Shape ret = shape;
  Layout* layout = ret.mutable_layout();
  for (int i = shape.dimensions_size() - 1; i >= 0; i--) {
    layout->add_minor_to_major(i);
  }
  return ret;
}

inline Shape ToLTCShape(std::vector<int64_t> shape, mnm::DType dtype) {
  return WithDefaultMinorToMajor(Shape(dtype, lazy_tensors::Span<const int64>(shape)));
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

inline DType ToMNMDType(const PrimitiveType type) {
  switch (type) {
    case PrimitiveType::S8: return DType(DTypeCode::kInt(), 8);
    case PrimitiveType::S64: return DType(DTypeCode::kInt(), 64);
    case PrimitiveType::PRED: return DType(DTypeCode::kUInt(), 1);
    case PrimitiveType::U8: return DType(DTypeCode::kUInt(), 8);
    case PrimitiveType::F16: return DType(DTypeCode::kFloat(), 16);
    case PrimitiveType::F32: return DType(DTypeCode::kFloat(), 32);
    case PrimitiveType::F64: return DType(DTypeCode::kFloat(), 64);
  }
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline std::tuple<std::vector<int64_t>, DType> ToMNMShape(const lazy_tensors::client::ShapeData& shape) {
  return std::make_tuple(shape.dimensions(), ToMNMDType(shape.element_type()));
}

inline std::tuple<std::vector<int64_t>, DType> ToMNMShape(const Shape& shape) {
  lazy_tensors::Span<const int64> dimension = shape.dimensions();
  return std::make_tuple(
    std::vector<int64_t>(dimension.begin(), dimension.end()),
    ToMNMDType(shape.element_type()));
}

inline Type ToMNMType(const Shape& shape) {
  std::vector<int64_t> vec_shape;
  DType dtype;
  Array<tvm::PrimExpr> arr_shape;
  std::tie(vec_shape, dtype) = ToMNMShape(shape);
  for (const auto& x : vec_shape) {
    arr_shape.push_back(Integer(x));
  }
  return TensorType(arr_shape, DataType(dtype.operator DLDataType()));
}

inline mnm::Device ToMNMDevice(const std::string& device) {
  // TODO(@hzfan): map LTC device to meta device
  return mnm::Device(DevType::kCPU(), 0);
}

inline std::tuple<Var, Var> PromoteDType(const Var& op0, const Var& op1) {
  using namespace mnm::binding;
  auto tty0 = Downcast<TensorType>(op0->checked_type());
  auto tty1 = Downcast<TensorType>(op1->checked_type());
  LTC_CHECK_EQ(tty0->dtype.lanes(), tty1->dtype.lanes());
  if (tty0->dtype.code() != tty1->dtype.code()) {
    if (tty0->dtype.is_float() && tty1->dtype.is_int()) {
      return std::make_tuple(op0, BindSymbol(mnm::ir::Call(Op::Get("mnm.op.cast_like"), {op1, op0})));
    } else if (tty1->dtype.is_float() && tty0->dtype.is_int()) {
      return std::make_tuple(BindSymbol(mnm::ir::Call(Op::Get("mnm.op.cast_like"), {op0, op1})), op1);
    }
    LTC_LOG(FATAL) << "Not implemented yet.";
  }
  if (tty0->dtype.bits() == tty1->dtype.bits()) {
    return std::make_tuple(op0, op1);
  } else if (tty0->dtype.bits() < tty1->dtype.bits()) {
    return std::make_tuple(BindSymbol(mnm::ir::Call(Op::Get("mnm.op.cast_like"), {op0, op1})), op1);
  } else {
    return std::make_tuple(op0, BindSymbol(mnm::ir::Call(Op::Get("mnm.op.cast_like"), {op1, op0})));
  }
}

}  // namespace mnm_backend
}  // namespace compiler
}  // namespace torch_lazy_tensors
