<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Add an Operator

This article introduces the process of adding a new operator to Ratex. It includes 7 possible steps: 

1. Adding operator (if applicable)
2. Turning off CPU fallback (if applicable).
3. Writing unit test.
4. Declaring operator. 
5. Extracting parameters to RAF suitable format.
6. Defining inference type (if applicable).
7. Adding custom autograd (if applicable)

We will use example operator implementations/lowering walkthroughs to help you better understand the process.


## Before You Start: Confirm that your operator is defined in RAF

Under `/third_party/raf/scripts/src_codegen/def_op.py` search for your operator and confirm your operator's implementation exists. If your operator does not exist in RAF, you will first need to implement the operator in RAF. Refer to [Add an Operator in RAF](https://github.com/awslabs/raf/blob/main/docs/wiki/3_dev_guide/Add-Operator.md).

## 1. Adding operator
Make sure to build Retex before proceeding otherwise some files (i.e., `RegisterLazy.cpp/LazyNativeFunctions.h`) will not be available. 
Operators are often decomposed into smaller opertors or do not exist in Ratex yet. First check `/ratex/csrc/RegisterLazy.cpp` & `/ratex/crsc/LazyNativeFunctions.h` to determine if your OP is already registered inside of Ratex. If you see your operator here, go ahead and move to step 2. If you do not see your operator here, you will need to add it in the `raf_native_functions.yaml` file. Refer to the [native list of operators](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml) for naming conventions/types/signatures. Upon adding your operator in the list, run `bash ./scripts/generate_code.sh` to have binding code generated for your operator in `RegisterLazy.cpp` & `LazyNativeFunctions.h` file. Next write your operator declaration in `/ratex/csrc/aten_raf_type.cpp`. This file is responsible for bridging to LazyTensor by converting Aten::Tensor to LazyTensor data structure, and then returning Aten::Tensor from DSC LazyTensor result. Keep in mind when using LazyTensor data structure that the values are not materialized meaning you cannot access values within the tensor.

Following this you will create an entry in the `ratex/lazy_tensor_core/csrc/tensor.h` for your newly added operator - model off exisiting implementations to define parameters/conversion. You will define behavior of your operator in `ratex/lazy_tensor_core/csrc/tensor_methods.cpp` which is used to create IR nodes. You will create IR node files for your operators as well, these are a simple inheritance based class structure. Utilize the following template to create your node.

```cpp
#pragma once

#include "lazy_tensor_core/csrc/ir.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class YourOperator : public Node {
 public:
  YourOperator((const Value& TensorInputs), (std::vector/int/double/etc OtherInput));

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  const std::vector/int/double/etc otherinput() const {
    return otherinput_;
  }


 private:
  std::vector/int/double/etc otherinput_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
```

Simply replace `YourOperator` with your operator's name, provide tensor values like TensorInput, and provide all other types (OtherInputs) normally.

```cpp
#include "lazy_tensor_core/csrc/ops/YourOperator.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

YourOperator::YourOperator(const ir::Value TensorInputs, int64_t OtherInputs)
    : Node(ir::OpKind(at::aten::YourOperator), {TensorInputs},
           num_outputs, lazy_tensors::util::MHash(OtherInputs)),
      otherinput_(OtherInputs) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr YourOperator::Clone(OpList operands) const {
  return MakeNode<YourOperator>(operands, otherinput_);
}

std::string YouOperator::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", otherinput=" << otherinput_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
```

Again, simply replace `YourOperator` with the name of your operator and provide Tensor values like `TensorInputs` while providing other values normally. Remember to refer to RAF & verify number of outputs (i.e., if RAF returns a tuple of 3, num_output should equal 3). The other inputs you might be concerned with saving are parameters required by RAF. You will be able to extract & convert them using this node object in `raf_node_lowering`. If your node is an aten node, then simply place your operator's files in `/ratex/lazy_tensor_core/csrc/ops` directory.

In the case you are adding an operator not a part of the aten library, your will need to create a custom node. To add a custom node, place the files under `/ratex/csrc/ops` directory. Next modify `/ratex/csrc/raf_ops.h/cpp` and add your custom operator name. Finally in your operator's cpp file modify the following:
`#include "ratex/csrc/ops/raf_ops.h"`
Change `Node(ir::OpKind(at::aten::YourOperator)` To `Node(your_raf_op)`

Remember if you are creating a custom node, you do not get flexibility of PyTorch flexibility work for you. PyTorch does not know about your custom operator, so it does not know how to decompose/bridge/autograd for you. Thus, if you are implementing a forward operator, make sure to add corresponding backward operator as well.

Finally, return to `ratex/lazy_tensor_core/csrc/tensor_methods.cpp` and include  your `.cpp/.h` files & finish IR node creation. Create an IR node as such:

```cpp
return InputTensor.CreateFrom(ir::ops::YourOperator(Tensors.GetIrValue(), other_inputs));
```
Basically provide inputs to your newly created class files to construct and IR node. You can refer to other similiar operators to deal with cases such as tuples, converions, etc.

You can now skip to step 3.

## 2. Turn Off CPU Fallback

If you operator is already registered, it might be possible that it falls back to CPU & does not have lowering. 
For missing operators in RAF, Ratex automatically falls back to sequential implementations on CPU (this can also be manually specified using `FALLBACK_ATEN_OP`). To make sure the new operator in Ratex can be lowered to RAF, we will need to turn off the CPU fallback. 

Take a look at the file `/ratex/csrc/aten_raf_type.cpp` and find your operator. In the case there is no lowering, the file will call fallback (sequential/CPU) implementations for operators. Look at other methods for reference.
We will use the operator `norm` as an example here.

The norm's default fallback implementations appear as:

```cpp
at::Tensor LazyNativeFunctions::norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                                     at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
  return AtenRAFTypeDefault::norm(self, p, dim, keepdim, dtype);
}

at::Tensor LazyNativeFunctions::norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                                     at::IntArrayRef dim, bool keepdim) {
  return AtenRAFTypeDefault::norm(self, p, dim, keepdim);
}

at::Tensor LazyNativeFunctions::norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                                     at::ScalarType dtype) {
  return AtenRAFTypeDefault::norm(self, p, dtype);
}

at::Tensor LazyNativeFunctions::norm(const at::Tensor& self, const at::Scalar& p) {
  return AtenRAFTypeDefault::norm(self, p);
}
```

Note that the code snippet above cannot be found in the source code. 
This is due to the reason that we have already added the LazyTensor support to the `norm` operator. 
We present the old code here for the illustration purpose.

As you can see, we are calling the fallback implementation in the form of `AtenRAFTypeDefault::norm()`, which 
is the native torch implementation. The CPU fallback implementations are defined in `/ratex/csrc/aten_cpu_fallback.h` and `/ratex/csrc/aten_cpu_fallback.cpp` (which are also removed in the current codebase due to the added support to norm).

Next, we will modify the above fallback implementations and redirect them to LazyTensor implementations, as shown below:

```cpp
at::Tensor LazyNativeFunctions::norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                                     at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::norm(self_tensor, p, dtype, dim, keepdim));
}

at::Tensor LazyNativeFunctions::norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                                     at::IntArrayRef dim, bool keepdim) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::norm(self_tensor, p, self_tensor.dtype(), dim, keepdim));
}

at::Tensor LazyNativeFunctions::norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                                     at::ScalarType dtype) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(LazyTensor::norm(
      self_tensor, p, dtype, lazy_tensors::util::Iota<int64_t>(self_tensor.shape().get().rank()),
      /*keep_reduced_dimensions=*/false));
}

at::Tensor LazyNativeFunctions::norm(const at::Tensor& self, const at::Scalar& p) {
  LTC_FN_COUNTER("raf::");
  LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
  return bridge::AtenFromLtcTensor(
      LazyTensor::norm(self_tensor, p, self_tensor.dtype(),
                       lazy_tensors::util::Iota<int64_t>(self_tensor.shape().get().rank()),
                       /*keep_reduced_dimensions=*/false));
}
```

Let's take a closer look at the code above. 
First, for each operator method, we first locate the corresponding LazyTensor method from 
`/ratex/lazy_tensor_core/csrc/tensor.h`. If your operator is new and not defined already, go ahead add in your operator signature here. Next in `/ratex/lazy_tensor_core/csrc/tensor_methods.cpp` define behavior for your method. The `tensor_methods` file is primarily used for IR node creation for you operator. Make sure to create corresponding `cpp/h` files for your operator & import them.
For example, for the operator `norm`, we find the LazyTensor implementation:
```c++
static LazyTensor norm(const LazyTensor& input, const c10::optional<at::Scalar>& p,
                       c10::optional<at::ScalarType> dtype, at::IntArrayRef dim, bool keepdim);
```                         

As you can see, the first native function:

```c++
at::Tensor LazyNativeFunctions::norm(const at::Tensor& self, const c10::optional<at::Scalar>& p,
                                     at::IntArrayRef dim, bool keepdim, at::ScalarType dtype)
```

and the LazyTensor function share some common parameters including
`p`, `dim`, and `keepdim`, and `dtype`. Meanwhile, LazyTensor function expects a `LazyTensor input` as input, we 
need to convert our original parameter `const at::Tensor self` to a LazyTensor.

To accomplish this, we call

```c++
LazyTensor self_tensor = bridge::raf_backend::GetLtcTensor(self);
```

to convert an `at::tensor` to `LazyTensor`.

| Input      | Expected   | Conversion                                |
|------------|------------|-------------------------------------------|
| at::Tensor | LazyTensor | bridge::raf_backend::GetLtcTensor(Input) |

and we call

```c++
bridge::AtenFromLtcTensor()
```

to convert a `LazyTensor` back to `at::tensor`.

For the rest native functions, we will need to add the missing operators manually.
Let's take a look at the 4th signature of norm's native functions:

```c++
at::Tensor LazyNativeFunctions::norm(const at::Tensor& self, const at::Scalar& p)
```

We are only provided 2 of the 5 required arguments to the LazyTensor function.
In this case, we can analyze the codebase for already existing implementations of parameter lowering (for example, `dim` is used in many operators and has generic conversion implementation already) or write the corresponding conversion. Here is how we fill gaps for norm:

| Type            | Param      | Generic implementation                    |
|-----------------|------------|-------------------------------------------|
| at:ScalarType   | dtype      | self_tensor.dtype()                       |
| at::IntArrayRef | dim        | lazy_tensors::util::Iota<int64_t>(self_tensor.shape().get().rank())                      |
| bool            | keepdim    | false                    |

As you can see, we try to follow torch in default ways of inference & parameters.

Finally, check `/ratex/csrc/RegisterLazy.cpp` (This file is autogenerated after compiling Ratex) to see if there is any wrapper processing that you need to do before calling method in `aten_raf_type.cpp`. Norm does not require any wrapper specific processing, so it simply just calls the methods. In the case there is a mismatch in parameters, you will need to handle conversions appropriately. 

Once you have this conversion completed and working correctly, delete the default CPU fallback implementations for 
your operator in `/ratex/csrc/aten_cpu_fallback.h` and `/ratex/csrc/aten_cpu_fallback.cpp`.

Note that if the operator you are adding has already been lowered to LazyTensor implementations in `aten_raf_type.cpp` at the beginning, you could skip this step.

## 3. Define Unit Tests

It is recommended to write a unit test immediately and view error stack trace. This is because some operators are converted to others implicitly (i.e. chunk defaults to split since split is more powerful), or may not require all implementation steps. Process the remaining guide based on the errors you are seeing. For the remaining guide, we will be using `stack` operator as a running example. Process this guide according to the errors you are seeing.

Under `tests/python/op` choose/create your corresponding test file. To determine which type your test falls under search `/third_party/raf/src/op/declare` for your operator. For stack, it falls under transform, so under `test_transform.py` implement unit test. Follow convention when declaring test: `test_<your-op>`. Use `pytest.mark.parametrize` to organize & iterate through the various testing/input conditions. For example, for a 2-dimensional tensor you can stack it row-wise (dim = 0) or column-wise (dim = 1) as well as stack different data types. The following test will run 4 tests:
- dim=0, dtype=float16 
- dim=0, dtype=float32
- dim=1, dtype=float16
- dim=1, dtype=float32

Make sure that all possible combinations are accounted for and configurations make sense. 

Be sure to refer to existing implementations that are most similar to yours and also utilize `/testing/common.py` to test you operator.

### Here is the stack test:

```python
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_stack(dim, dtype):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *tensors):
            return torch.stack(tuple(tensors), dim)

    shape = [3, 4]
    x = torch.randn(*shape).to(dtype)
    y = torch.randn(*shape).to(dtype)
    z = torch.randn(*shape).to(dtype)

    verify_step(Model(), [x, y, z], jit_script=False)
```
### Compile and run your tests

If you modify any of the cpp files or python files in the package (such as common.py), you will need to recompile.
Under root directory (ratex) run:

```
bash ./scripts/build_ratex.sh
```

Resolve any compilation errors, afterwards under `tests/python/op` run: ```pytest <your-test-file>.py -k 'test_<your-op>'```

### Error output

<details> <summary> The following is the error stack trace we receive from executing stack test case. </summary>
<blockquote>
E       RuntimeError: [18:22:09] /workspace/ratex/csrc/compiler/raf_node_lowering.cpp:1184: Shape inference not supported for operator: aten::stack
E       Stack trace:
E         0: Infer
E               at /workspace/ratex/csrc/compiler/raf_node_lowering.cpp:1184
E         1: operator()
E               at /workspace/ratex/lazy_tensor_core/csrc/ops/stack.cpp:21
E         2: _M_invoke
E               at /usr/include/c++/7/bits/std_function.h:302
E         3: std::function<lazy_tensors::Shape ()>::operator()() const
E               at /usr/include/c++/7/bits/std_function.h:706
E         4: torch_lazy_tensors::ir::Node::GetOpShape(std::function<lazy_tensors::Shape ()> const&) const
E               at /workspace/ratex/lazy_tensor_core/csrc/ir.cpp:261
E         5: torch_lazy_tensors::ir::Node::SetShapeDeferred(std::function<lazy_tensors::Shape ()> const&)
E               at /workspace/ratex/lazy_tensor_core/csrc/ir.cpp:176
E         6: torch_lazy_tensors::ir::ops::Stack::Stack(lazy_tensors::Span<torch_lazy_tensors::ir::Value const>, long)
E               at /workspace/ratex/lazy_tensor_core/csrc/ops/stack.cpp:21
E         7: void __gnu_cxx::new_allocator<torch_lazy_tensors::ir::ops::Stack>::construct<torch_lazy_tensors::ir::ops::Stack, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&>(torch_lazy_tensors::ir::ops::Stack*, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&)
E               at /usr/include/c++/7/ext/new_allocator.h:136
E         8: void std::allocator_traits<std::allocator<torch_lazy_tensors::ir::ops::Stack> >::construct<torch_lazy_tensors::ir::ops::Stack, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&>(std::allocator<torch_lazy_tensors::ir::ops::Stack>&, torch_lazy_tensors::ir::ops::Stack*, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&)
E               at /usr/include/c++/7/bits/alloc_traits.h:475
E         9: std::_Sp_counted_ptr_inplace<torch_lazy_tensors::ir::ops::Stack, std::allocator<torch_lazy_tensors::ir::ops::Stack>, (__gnu_cxx::_Lock_policy)2>::_Sp_counted_ptr_inplace<std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&>(std::allocator<torch_lazy_tensors::ir::ops::Stack>, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&)
E               at /usr/include/c++/7/bits/shared_ptr_base.h:526
E         10: std::__shared_count<(__gnu_cxx::_Lock_policy)2>::__shared_count<torch_lazy_tensors::ir::ops::Stack, std::allocator<torch_lazy_tensors::ir::ops::Stack>, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&>(std::_Sp_make_shared_tag, torch_lazy_tensors::ir::ops::Stack*, std::allocator<torch_lazy_tensors::ir::ops::Stack> const&, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&)
E               at /usr/include/c++/7/bits/shared_ptr_base.h:637
E         11: std::__shared_ptr<torch_lazy_tensors::ir::ops::Stack, (__gnu_cxx::_Lock_policy)2>::__shared_ptr<std::allocator<torch_lazy_tensors::ir::ops::Stack>, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&>(std::_Sp_make_shared_tag, std::allocator<torch_lazy_tensors::ir::ops::Stack> const&, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&)
E               at /usr/include/c++/7/bits/shared_ptr_base.h:1295
E         12: std::shared_ptr<torch_lazy_tensors::ir::ops::Stack>::shared_ptr<std::allocator<torch_lazy_tensors::ir::ops::Stack>, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&>(std::_Sp_make_shared_tag, std::allocator<torch_lazy_tensors::ir::ops::Stack> const&, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&)
E               at /usr/include/c++/7/bits/shared_ptr.h:344
E         13: std::shared_ptr<torch_lazy_tensors::ir::ops::Stack> std::allocate_shared<torch_lazy_tensors::ir::ops::Stack, std::allocator<torch_lazy_tensors::ir::ops::Stack>, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&>(std::allocator<torch_lazy_tensors::ir::ops::Stack> const&, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&)
E               at /usr/include/c++/7/bits/shared_ptr.h:691
E         14: std::shared_ptr<torch_lazy_tensors::ir::ops::Stack> std::make_shared<torch_lazy_tensors::ir::ops::Stack, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&>(std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&)
E               at /usr/include/c++/7/bits/shared_ptr.h:707
E         15: std::shared_ptr<torch_lazy_tensors::ir::Node> torch_lazy_tensors::ir::MakeNode<torch_lazy_tensors::ir::ops::Stack, std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&>(std::vector<torch_lazy_tensors::ir::Value, std::allocator<torch_lazy_tensors::ir::Value> >&, long&)
E               at /workspace/ratex/lazy_tensor_core/csrc/ir.h:333
E         16: torch_lazy_tensors::LazyTensor::stack(lazy_tensors::Span<torch_lazy_tensors::LazyTensor const>, long)
E               at /workspace/ratex/lazy_tensor_core/csrc/tensor_methods.cpp:2054
E         17: torch_lazy_tensors::LazyNativeFunctions::stack(c10::ArrayRef<at::Tensor>, long)
E               at /workspace/ratex/csrc/aten_raf_type.cpp:2483
E         18: wrapper__stack
E               at /workspace/ratex/csrc/RegisterLazy.cpp:1896
E         19: operator()
E               at /usr/local/lib/python3.7/dist-packages/torch/include/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h:13
E         20: call
E               at /usr/local/lib/python3.7/dist-packages/torch/include/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h:446
E         21: at::_ops::stack::redispatch(c10::DispatchKeySet, c10::ArrayRef<at::Tensor>, long)
E         22: torch::autograd::VariableType::(anonymous namespace)::stack(c10::DispatchKeySet, c10::ArrayRef<at::Tensor>, long)
E         23: _ZN3c104impl28wrap_kernel_functor_unboxed_INS0_6detail24WrapFunctionInt
E         24: at::_ops::stack::call(c10::ArrayRef<at::Tensor>, long)
E         ...
</blockquote>
</details>
<br>

As you can see, Ratex is complaining about shape inference. The reason we need to support shape inference goes back to the basic logic of compiled vs interpreted languages. Interpreted languages (i.e., sequential PyTorch in this case) dynamically check for and use size/type on the fly. However, in compiled languages, compilers often apply optimization. To apply optimization, the compiler must know exactly the expected inputs/outputs shapes & data types (this is vital for operator fusion for example) so that it can make sketches/graphs to execute. We must explicitly tell the compiler the shape/sizes, and we do this using the `Infer` method. The Infer method allows Ratex to define the output type (shape and data) of the operator. Before we define the Infer method, we need to declare & build conversion in the next step.

## 4. Declare an Operator

In `/ratex/csrc/compiler/raf_node_lowering.cpp`, under private of ```class RAFNodeLowering```, write the corresponding declare statement. 

To decide between `DECLARE_OP` vs `DECLARE_OP2`, determine if your operator is contained in and or is a composition of operators inside `/ratex/lazy_tensor_core/csrc/ops/ops.cpp`. If you operator is contained/aggregate within the `ops.cpp` file - then your operator is `DECLARE_OP`. Otherwise, your operator is inside its own file (`/ratex/lazy_tensor_core/csrc/ops/<yourop>.cpp/h` or `/ratex/csrc/ops/<yourop>.cpp/h`) and is `DECLARE_OP2`. If an already existing file of your operator is not found, you will need to create the corresponding cpp/h file for your operator (model off similar existing operator implementation). Make sure to `#include` your operators `.cpp/.h` files in `raf_node_lowering.cpp`.

Since stack is stored as `/ratex/lazy_tensor_core/csrc/ops/stack.cpp/h` we write:

```DECLARE_OP2(Stack);```

If you operator cannot be inferred (check unit test stack trace), also declare Shape signature. For stack, we did get a shape inference error, so we will declare it.

```lazy_tensors::Shape InferStack(const ir::ops::Stack* node);```

Under `Var RAFNodeLowering::LowerToRAF`:

```HANDLE_GENERIC_OP2(Stack, at::aten::stack)```


## 5. Define Conversion

After the declaration of lowering functions in the previous step, we will work on implement the function definitions.
The following is Stack implementation in the `/ratex/csrc/compiler/raf_node_lowering.cpp`:

```cpp
Var BuildStack(const std::vector<Var>& ops, const ir::ops::Stack* node) {
  std::vector<Expr> ops_expr(ops.begin(), ops.end());
  Var x = BindSymbol(raf::ir::Tuple(Array<Expr>(ops_expr)));
  Expr axis = MakeConstant(Int(node->dim()));
  return BindSymbol(raf::ir::Call(Op::Get("raf.op.stack"), {x, axis}));
}

Var RAFNodeLowering::LowerStack(const ir::ops::Stack* node) {
  std::vector<Var> ops;
  for (const auto& op : node->operands()) ops.push_back(loctx()->GetOutputOp(op));
  return BuildStack(ops, node);
}
```

As you can see, since stack takes in a tuple of tensors to concatenate, we have a traversal - this may not apply to your operator. Also, take a look at `stack.h`, we can see dimension (dim) is stored as private variable, so we extract this separate from ops. 

`Var` & `Expr` are RAF-specific types - since we are lowering PyTorch operators to RAF we need to convert appropriately. 

For adding new operators, find an already existing implementation that is most similar to the operator you are trying to implement. Refer to `/ratex/lazy_tensor_core/csrc/ops/<yourop>.cpp/h` for implementation details & operators to extract. 

## 6. Define Inference (if applicable)

Under `lazy_tensors::Shape RAFNodeLowering::Infer` add your operator case.

```cpp
case at::aten::stack: {
    return InferStack(ir::NodeCast<ir::ops::Stack>(node, ir::OpKind(at::aten::stack)));
}
```

Now define the Infer, try to model after similar existing implementations. `LTC_CHECK_EQ` is just sanity check to confirm number of operators.

```cpp
lazy_tensors::Shape RAFNodeLowering::InferStack(const ir::ops::Stack* node) {
  LTC_CHECK_EQ(node->operands().size(), 1U);
  std::vector<Var> ops;
  for (const auto& x : node->operands()) {
    ops.push_back(MakeVar("operand", ToRAFType(x.shape())));
  }
  Var out = BuildStack(ops, node);
  Expr body = InferType(ExtractBinding(out, ops));
  return ToLTCShape(body->checked_type());
}
```

## 7. Define custom autograd (if applicable)

If your operator requires some custom functionality in relation to automatic differentiation, then you will need to register the operator as a custom autograd operator. For this example, we will utilize dropout.

First in `raf_native_functions.yaml` put operator under autograd header. Next in inside 
`/ratex/csrc/aten_raf_type.cpp` for your operator write the corresponding apply method.

```cpp
at::Tensor LazyNativeFunctions::dropout(const at::Tensor& input, double p, bool train) {
  return aten_autograd_ops::Dropout::apply(input, p, train);
}
```

`/ratex/csrc/aten_autograd_ops.h/cpp` add your corresponding signatures for your operator. Ensure you provide identical signatures to to `aten_raf_type` with the addition of `ctx` variable. Providing extra arguments or changing the signature will cause the functionality to break.

The following is the implementation for dropout:
`aten_autograd_ops.h`
```cpp
struct Dropout : public torch::autograd::Function<Dropout> {
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& input,
                               double p, bool train);
  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_output);
};
```

`aten_autograd_ops.cpp`

```cpp
torch::Tensor Dropout::forward(torch::autograd::AutogradContext* ctx, const at::Tensor& input,
                               double p, bool train) {
  ctx->saved_data["p"] = p;
  auto outputs = LazyTensor::dropout(bridge::GetLtcTensor(input), p, train);
  ctx->save_for_backward({bridge::AtenFromLtcTensor(std::get<1>(outputs)),
                          bridge::AtenFromLtcTensor(std::get<2>(outputs))});
  return bridge::AtenFromLtcTensor(std::get<0>(outputs));
}

torch::autograd::variable_list Dropout::backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grad_output) {
  auto p = ctx->saved_data["p"];

  auto saved = ctx->get_saved_variables();
  auto mask = bridge::GetLtcTensor(saved[0]);
  auto reserved_space = bridge::GetLtcTensor(saved[1]);
  auto grad = bridge::GetLtcTensor(grad_output[0]);
  auto results = LazyTensor::dropout_backward(grad, mask, reserved_space);

  auto grad_outputs = bridge::AtenFromLtcTensor(results);
  torch::Tensor undef;
  torch::autograd::variable_list grad_result = {grad_outputs, undef, undef};

  return grad_result;
}
```

As you can see, we simply move `aten_raf_type` code to this file and call `tensor_methods.cpp` like normal. Thus, even for your autograd backward operator, you will need to conduct the entire process of adding a regular operator. However, the powerful functionality of autograd ops is the ability to save forward pass varaibles (inputs & outputs) for backward pass. So we can save forward pass input variables like p for backward. However, we can also store outputs. For example, we lower dropout here to RAF dropout. RAF dropout outputs a tuple of 3 tensors: output, mask, reserve_space. PyTorch only requires output tensor, however, RAF dropout backward requires gradient, mask, and reserve_space. As you can see, utilizing the flexibility of the aten_autograd_ops we can meet domain specific compiler demands. 


# Contribute Your Operators

If you would like to commit/contribute, please run formatters to properly format your code (GitHub jobs will fail without). Run the following commands under root directory. 
<br>
`bash scripts/lint/git-clang-format.sh -i upstream/main` This auto-formats the cpp files.
<br>
`bash scripts/lint/git-black.sh -i upstream/main` This auto-formats the Python files.
<br>
`bash scripts/lint/check-lint.sh` This checks more python syntax, fix if you see any warning.
<br>
