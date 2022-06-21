<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Overlap Trace and Execution

In ratex, operators are lazy: `loss = model(input)` does not execute the forward operation immediately. Instead, the operation is traced (recorded) in a computation graph. It will not be executed until its output is needed or an explicit barrier `mark_step()` is called.

A typical training loop can be divided into two parts: trace and execution. Although trace is fast compared to execution, it still takes few milliseconds in each iteration, so we make the execution asynchronies to hide the trace latency of the next iteration. This is called "overlap trace (of iteration n+1) and execution (of iteration n)".

## Training Loop Sample

```python
losses = []
for train_x, train_label in enumerate(train_loader):
    # ...
    t_optimizer.zero_grad()
    output = t_model(train_x)  # fwd
    t_loss = loss_fn(output)  # loss
    t_loss.backward()  # bwd
    t_optimizer.step()  # update
    mark_step()  # barrier
    losses.append(t_loss.item())  # print loss
print(losses)
```

This is a bad example - Trace and execution takes place sequentially, which makes the iteration slower. The problem lies at `t_loss.item()`:
  1. `output = t_model(train_x)` traces the forward graph.
  2. `t_loss = loss_fn(output)` traces the loss function.
  3. `t_loss.backward()` traces the backward function.
Until here, no execution takes place.
  4. `t_optimizer.step()` traces the optimizer graph
  5. `mark_step()` executes all traced graph asynchronously.
  6. `losses.append(t_loss.item())` waits for the value of `t_loss` to be ready.
  
To meet the semantic of step 6 (print the value of `t_loss`), all executions have to be done at that line, which creates a synchronous barrier and prevents the trace of next iteration from happening simultaneously.

## Overlap

```python
losses = []
for train_x, train_label in enumerate(train_loader):
    # ...
    t_optimizer.zero_grad()
    output = t_model(train_x)  # fwd
    t_loss = loss_fn(output)  # loss
    t_loss.backward()  # bwd
    t_optimizer.step()  # update
    mark_step()  # barrier
    losses.append(t_loss)  # don't print the value here to keep it lazy.
print([for loss.item() in losses])
```

Trace and execution overlaps in this example. The key is `losses.append(t_loss)`. It does not block the loop until `t_loss` is ready. Trace in the next iteration overlaps with the execution in the current iteration. If execution takes longer than trace (which is typically the case), trace can be completely hidden by execution.
