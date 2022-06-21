/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ratex/csrc/compiler/raf_lowering_context.h"
#include "client/base_computation_client.h"

namespace torch_lazy_tensors {
namespace serialization {

using namespace tvm;
using namespace raf::ir;

using LTCShape = lazy_tensors::Shape;
using LTCProgramShape = lazy_tensors::ProgramShape;
using LTCGenericComputationRAF = compiler::raf_backend::GenericComputationRAF;
using LTCComputation = lazy_tensors::ComputationClient::Computation;
using LTCBaseComputation = ratex::BaseComputationClient::BaseComputation;
using lazy_tensors::PrimitiveType;

class GenericComputationRAFNode : public Object {
 public:
  /*! \brief the raf function to be compiled */
  Expr computation;
  /*! \brief the parameters of computation_ that represent model states */
  Array<Var> model_states;
  /*! \brief maps input to output if they are aliased */
  Map<Integer, Integer> alias;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("computation", &computation);
    v->Visit("model_states", &model_states);
    v->Visit("alias", &alias);
  }

  bool SEqualReduce(const GenericComputationRAFNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(computation, other->computation) && equal(model_states, other->model_states) &&
           equal(alias, other->alias);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(computation);
    hash_reduce(model_states);
    hash_reduce(alias);
  }

  static constexpr const char* _type_key = "ratex.GenericComputationRAF";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(GenericComputationRAFNode, Object);
};

class GenericComputationRAF : public ObjectRef {
 public:
  TVM_DLL GenericComputationRAF(LTCGenericComputationRAF);

  operator LTCGenericComputationRAF() const;

  TVM_DEFINE_OBJECT_REF_METHODS(GenericComputationRAF, ObjectRef, GenericComputationRAFNode);
};

class ShapeNode : public Object {
 public:
  Integer element_type;
  Array<Integer> dimensions;
  Array<ObjectRef> element_shapes;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("element_type", &element_type);
    v->Visit("dimensions", &dimensions);
    v->Visit("element_shapes", &element_shapes);
  }

  bool SEqualReduce(const ShapeNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(element_type, other->element_type) && equal(dimensions, other->dimensions) &&
           equal(element_shapes, other->element_shapes);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(element_type);
    hash_reduce(dimensions);
    hash_reduce(element_shapes);
  }

  static constexpr const char* _type_key = "ratex.Shape";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(ShapeNode, Object);
};

class Shape : public ObjectRef {
 public:
  TVM_DLL Shape(LTCShape);

  operator LTCShape() const;

  TVM_DEFINE_OBJECT_REF_METHODS(Shape, ObjectRef, ShapeNode);
};

class ProgramShapeNode : public Object {
 public:
  Array<Shape> parameters;
  Array<String> parameter_names;
  Shape result;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("parameters", &parameters);
    v->Visit("parameter_names", &parameter_names);
    v->Visit("result", &result);
  }

  bool SEqualReduce(const ProgramShapeNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(parameters, other->parameters) && equal(parameter_names, other->parameter_names) &&
           equal(result, other->result);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(parameters);
    hash_reduce(parameter_names);
    hash_reduce(result);
  }

  static constexpr const char* _type_key = "ratex.ProgramShape";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(ProgramShapeNode, Object);
};

class ProgramShape : public ObjectRef {
 public:
  TVM_DLL ProgramShape(LTCProgramShape);

  operator LTCProgramShape() const;

  TVM_DEFINE_OBJECT_REF_METHODS(ProgramShape, ObjectRef, ProgramShapeNode);
};

class ComputationNode : public Object {
 public:
  GenericComputationRAF computation;
  ProgramShape program_shape;
  Array<String> devices;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("computation", &computation);
    v->Visit("program_shape", &program_shape);
    v->Visit("devices", &devices);
  }

  bool SEqualReduce(const ComputationNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(computation, other->computation) && equal(program_shape, other->program_shape) &&
           equal(devices, other->devices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(computation);
    hash_reduce(program_shape);
    hash_reduce(devices);
  }

  static constexpr const char* _type_key = "ratex.Computation";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(ComputationNode, Object);
};

class Computation : public ObjectRef {
 public:
  TVM_DLL Computation(LTCComputation);

  operator LTCComputation() const;

  TVM_DEFINE_OBJECT_REF_METHODS(Computation, ObjectRef, ComputationNode);
};

class BaseComputationNode : public ComputationNode {
 public:
  Map<Integer, Integer> alias;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("alias", &alias);
    ComputationNode::VisitAttrs(v);
  }

  bool SEqualReduce(const BaseComputationNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(alias, other->alias) && ComputationNode::SEqualReduce(other, equal);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(alias);
    ComputationNode::SHashReduce(hash_reduce);
  }

  static constexpr const char* _type_key = "ratex.BaseComputation";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(BaseComputationNode, ComputationNode);
};

class BaseComputation : public Computation {
 public:
  TVM_DLL BaseComputation(LTCBaseComputation);

  operator LTCBaseComputation() const;

  TVM_DEFINE_OBJECT_REF_METHODS(BaseComputation, Computation, BaseComputationNode);
};

}  // namespace serialization
}  // namespace torch_lazy_tensors
