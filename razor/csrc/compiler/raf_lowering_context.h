/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_set>

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/lowering_context.h"

#include "raf/ir.h"
#include "raf/value.h"

namespace torch_lazy_tensors {
namespace compiler {
namespace raf_backend {

class GenericComputationRAF : public lazy_tensors::GenericComputation {
 public:
  GenericComputationRAF(
      raf::ir::Expr computation,
      const std::unordered_set<raf::ir::Var, tvm::ObjectPtrHash, tvm::ObjectPtrEqual>& model_states,
      const std::unordered_map<int64_t, int64_t>& alias)
      : computation_(computation), model_states_(model_states), alias_(alias) {
  }

  lazy_tensors::StatusOr<lazy_tensors::ProgramShape> GetProgramShape() const override;

  const raf::ir::Expr& computation() const {
    return computation_;
  }

  const std::unordered_set<raf::ir::Var, tvm::ObjectPtrHash, tvm::ObjectPtrEqual>& model_states()
      const {
    return model_states_;
  }

  const std::unordered_map<int64_t, int64_t>& alias() const {
    return alias_;
  }

 private:
  /*! \brief the raf function to be compiled */
  raf::ir::Expr computation_;
  /*! \brief the parameters of computation_ that represent model states */
  std::unordered_set<raf::ir::Var, tvm::ObjectPtrHash, tvm::ObjectPtrEqual> model_states_;
  /*! \brief maps input to output if they are aliased */
  std::unordered_map<int64_t, int64_t> alias_;
};

class RAFLoweringContext : public ir::LoweringContext {
 public:
  RAFLoweringContext(const std::string& name, Device device) : ir::LoweringContext(name, device) {
  }

  RAFLoweringContext(const std::string& name, Device device,
                     absl::Span<const ir::Node* const> post_order,
                     ir::Util::EmissionMap emit_status)
      : ir::LoweringContext(name, device, post_order, emit_status) {
    auto lowering = NodeLowering::Create(this);
    for (auto node : post_order) {
      bool ok = lowering->Lower(node);
      LTC_CHECK(ok) << "Failed to lower: " << *node;
    }
  }

  lazy_tensors::Shape GetResultShape(size_t index) const override;

  size_t AddResult(const ir::Output& output) override;

  lazy_tensors::StatusOr<std::shared_ptr<lazy_tensors::GenericComputation>> Build() override;

  void LowerNodeToResult(const ir::Node* node) override;

  // void AddParameter(const ir::Output& output, size_t index,
  //                   const lazy_tensors::Shape& shape,
  //                   const std::string& name) override;

  void SetUpAlias(const lazy_tensors::ShapeIndex& output_index, int64_t param_number,
                  const lazy_tensors::ShapeIndex& param_index) override;

  // Retrieves the lowered operation for a output. If the requested output is
  // not available yet, the graph behind the output's Node is lowered, and the
  // corresponding RAF operation returned.
  raf::ir::Var GetOutputOp(const ir::Output& output);

  // Assigns the given RAF operation to the specified output. As outputs are
  // lowered in a post-order fashion, later nodes should always find their
  // operands among the emitted outputs.
  void AssignOutputOp(const ir::Output& output, const raf::ir::Var& op);

  // If a parameter associated with data has already been declared, it will be
  // returned. Otherwise a new one will be created, associated with the tensor
  // held in data.
  raf::ir::Var GetParameter(const std::shared_ptr<lazy_tensors::client::Data>& data);

 private:
  struct Parameter {
    raf::ir::Var param;
    size_t index = 0;
  };

  // Adds the output of a given operation to the result tuple. Returns the index
  // of the output within the tuple.
  size_t AddResult(const raf::ir::Var& op);

  raf::ir::Var GetResult(size_t index) const;

  // Lowers a single IR node. All the inputs to the node must have a lowering
  // before calling this API. Returns the generated RAF operations.
  raf::ir::Var LowerNode(const ir::Node* node);

  // Get parameters
  std::vector<raf::ir::Var> GetParams() const;

  std::unordered_map<lazy_tensors::client::Data::OpaqueHandle, Parameter> parameters_map_;
  std::vector<raf::ir::Var> root_tuple_;
  ir::OutputMap<raf::ir::Var> emitted_outputs_;
  /*! \brief the parameters of computation_ that represent model states */
  std::unordered_set<raf::ir::Var, tvm::ObjectPtrHash, tvm::ObjectPtrEqual> model_states_;
  /*! \brief maps input to output if they are aliased */
  std::unordered_map<int64_t, int64_t> alias_;
};

raf::ir::Var LowerNodeToRAF(const ir::Node* node, RAFLoweringContext* loctx);

}  // namespace raf_backend
}  // namespace compiler
}  // namespace torch_lazy_tensors
