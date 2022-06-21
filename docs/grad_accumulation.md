<!--- Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Gradient Accumulation

Gradient accumulation is a technique to simulate large batch size training in memory-constrained and network-constrained system.
Gradient accumulation is used in training situation where there’s not enough memory and/or network bandwidth to perform large batch size needed for optimal convergence.

When using gradient accumulation, instead of updating weights after each forward-backward pass of a mini-batch, we store and accumulate their gradients from forward-backward passes of several mini-batches, and update weights once. In this way, we could achieve a similar behavior of training with a larger batch size and might help loss convergence.


## Usage in PyTorch

It's easy to use this technique in PyTorch by simply modifying several lines of a training loop.

**A typical training loop:**

```python
for train_x, train_label in enumerate(train_loader):
    # ...
    t_optimizer.zero_grad()
    output = t_model(train_x)  # fwd
    t_loss = loss_fn(output)  # loss
    t_loss.backward()  # bwd
    t_optimizer.step()  # update
```

**Use gradient accumulation:**

```python
t_optimizer.zero_grad()
for idx, (train_x, train_label) in enumerate(train_loader):
    # ...
    output = t_model(train_x)  # fwd
    t_loss = loss_fn(output) / accu_grad_steps  # loss
    t_loss.backward()  # bwd
    if idx % accu_grad_steps == 0:  # update
        t_optimizer.step()
        t_optimizer.zero_grad()
```

*Note that the modified loss has a scaling factor `1/accu_grad_steps`.*

## Usage in RATEX

As can be seen from the following example, when using gradient accumulation, the only thing we need to do is adding another mark step after backward:

```python
t_optimizer.zero_grad()
for idx, (train_x, train_label) in enumerate(train_loader):
    # ...
    output = t_model(train_x)  # fwd
    t_loss = loss_fn(output) / accu_grad_steps  # loss
    t_loss.backward()  # bwd
    lm.mark_step()  # mark the fwd/bwd graph
    if idx % accu_grad_steps == 0:  # update
        t_optimizer.step()
        t_optimizer.zero_grad()
        lm.mark_step()  # mark optimizer graph
```

In this way, one training loop will be seperated into two computation graphs in backend, one for forward-backward pass and one for optimizer. The optimizer graph will be invoked per `accu_grad_steps` iterations.
