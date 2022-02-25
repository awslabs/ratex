/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "razor/csrc/serialization/serialization.h"

#include "razor/csrc/serialization/utils.h"

namespace torch_lazy_tensors {
namespace serialization {

using namespace mnm::ir;

GenericComputationMNM::GenericComputationMNM(LTCGenericComputationMNM x) {
  ObjectPtr<GenericComputationMNMNode> n = make_object<GenericComputationMNMNode>();
  n->computation = x.computation();
  n->model_states = ToTVMFromLTC<Array<Var>>(x.model_states());
  n->alias = ToTVMFromLTC<Map<Integer, Integer>>(x.alias());
  data_ = std::move(n);
}

GenericComputationMNM::operator LTCGenericComputationMNM() const {
  return LTCGenericComputationMNM{
      get()->computation,
      ToLTCFromTVM<std::unordered_set<mnm::ir::Var, tvm::ObjectPtrHash, tvm::ObjectPtrEqual>>(
          get()->model_states),
      ToLTCFromTVM<std::unordered_map<int64_t, int64_t>>(get()->alias)};
}

TVM_REGISTER_NODE_TYPE(GenericComputationMNMNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<GenericComputationMNMNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const GenericComputationMNMNode*>(ref.get());
      p->stream << "GenericComputationMNMNode(" << node->computation << ", " << node->model_states
                << ", " << node->alias << ")";
    });

Shape::Shape(LTCShape x) {
  ObjectPtr<ShapeNode> n = make_object<ShapeNode>();
  n->element_type = ToTVMFromLTC<Integer>((int)x.element_type());
  n->dimensions = ToTVMFromLTC<Array<Integer>>(x.dimensions());
  n->element_shapes = ToTVMFromLTC<Array<ObjectRef>>(x.tuple_shapes());
  data_ = std::move(n);
}

Shape::operator LTCShape() const {
  std::vector<LTCShape> tuple_shapes = ToLTCFromTVM<std::vector<LTCShape>>(get()->element_shapes);
  std::vector<int64> dimensions = ToLTCFromTVM<std::vector<int64>>(get()->dimensions);
  return get()->element_type->value == (int)PrimitiveType::TUPLE
             ? LTCShape(tuple_shapes)
             : LTCShape(ToLTCFromTVM<PrimitiveType>(get()->element_type), dimensions);
}

TVM_REGISTER_NODE_TYPE(ShapeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShapeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ShapeNode*>(ref.get());
      p->stream << "ShapeNode(" << node->element_type << ", " << node->dimensions << ", "
                << node->element_shapes << ")";
    });

ProgramShape::ProgramShape(LTCProgramShape x) {
  ObjectPtr<ProgramShapeNode> n = make_object<ProgramShapeNode>();
  n->parameters = ToTVMFromLTC<Array<Shape>>(x.parameters());
  n->parameter_names = ToTVMFromLTC<Array<String>>(x.parameter_names());
  n->result = ToTVMFromLTC<Shape>(x.result());
  data_ = std::move(n);
}

ProgramShape::operator LTCProgramShape() const {
  return LTCProgramShape{ToLTCFromTVM<std::vector<LTCShape>>(get()->parameters),
                         ToLTCFromTVM<std::vector<std::string>>(get()->parameter_names),
                         ToLTCFromTVM<LTCShape>(get()->result)};
}

TVM_REGISTER_NODE_TYPE(ProgramShapeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ProgramShapeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ProgramShapeNode*>(ref.get());
      p->stream << "ProgramShapeNode(" << node->parameters << ", " << node->parameter_names << ", "
                << node->result << ")";
    });

Computation::Computation(LTCComputation x) {
  ObjectPtr<ComputationNode> n = make_object<ComputationNode>();
  n->computation = *static_cast<LTCGenericComputationMNM*>(x.computation());
  n->program_shape = x.program_shape();
  n->devices = ToTVMFromLTC<Array<String>>(x.devices());
  data_ = std::move(n);
}

Computation::operator LTCComputation() const {
  return LTCComputation{std::make_shared<LTCGenericComputationMNM>(
                            get()->computation.operator LTCGenericComputationMNM()),
                        ToLTCFromTVM<LTCProgramShape>(get()->program_shape),
                        ToLTCFromTVM<std::vector<std::string>>(get()->devices)};
}

TVM_REGISTER_NODE_TYPE(ComputationNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ComputationNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ComputationNode*>(ref.get());
      p->stream << "ComputationNode(" << node->computation << ", " << node->program_shape << ", "
                << node->devices << ")";
    });

BaseComputation::BaseComputation(LTCBaseComputation x) {
  ObjectPtr<BaseComputationNode> n = make_object<BaseComputationNode>();
  n->computation = *static_cast<LTCGenericComputationMNM*>(x.computation());
  n->program_shape = x.program_shape();
  n->devices = ToTVMFromLTC<Array<String>>(x.devices());
  n->alias = ToTVMFromLTC<Map<Integer, Integer>>(x.alias);
  data_ = std::move(n);
}

BaseComputation::operator LTCBaseComputation() const {
  return LTCBaseComputation(std::make_shared<LTCGenericComputationMNM>(
                                get()->computation.operator LTCGenericComputationMNM()),
                            ToLTCFromTVM<LTCProgramShape>(get()->program_shape),
                            ToLTCFromTVM<std::vector<std::string>>(get()->devices),
                            ToLTCFromTVM<std::unordered_map<int64_t, int64_t>>(get()->alias));
}

TVM_REGISTER_NODE_TYPE(BaseComputationNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BaseComputationNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const BaseComputationNode*>(ref.get());
      p->stream << "BaseComputationNode(" << node->computation << ", " << node->program_shape
                << ", " << node->devices << ", " << node->alias << ")";
    });

}  // namespace serialization
}  // namespace torch_lazy_tensors
