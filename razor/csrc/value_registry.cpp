/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tvm/runtime/data_type.h"
#include "tvm/runtime/ndarray.h"
#include "tvm/node/functor.h"
#include "tvm/ir/module.h"
#include "mnm/executor.h"
#include "mnm/ir.h"
#include "mnm/registry.h"
#include "mnm/tensor.h"
#include "mnm/value.h"

namespace mnm {
namespace value {

using namespace mnm::ir;

class ValueRegistry {
 public:
  std::mutex mu;
  std::unordered_map<const ValueObj*, int64_t> value_to_handle;
  std::vector<Value> handle_to_value;

  static ValueRegistry* Get() {
    static ValueRegistry* instance = new ValueRegistry();
    return instance;
  }
};

Integer ValueToHandle(Value value) {
  static ValueRegistry* mgr = ValueRegistry::Get();
  {
    std::lock_guard<std::mutex> lock(mgr->mu);
    if (mgr->value_to_handle.find(value.as<ValueObj>()) == mgr->value_to_handle.end()) {
      mgr->value_to_handle[value.as<ValueObj>()] = mgr->handle_to_value.size();
      mgr->handle_to_value.push_back(value);
    }
    return mgr->value_to_handle.at(value.as<ValueObj>());
  }
}

Value HandleToValue(Integer value) {
  static ValueRegistry* mgr = ValueRegistry::Get();
  {
    std::lock_guard<std::mutex> lock(mgr->mu);
    return mgr->handle_to_value[value->value];
  }
}

MNM_REGISTER_GLOBAL("mnm.value.ValueToHandle").set_body_typed(ValueToHandle);
MNM_REGISTER_GLOBAL("mnm.value.HandleToValue").set_body_typed(HandleToValue);

}  // namespace value
}  // namespace mnm
