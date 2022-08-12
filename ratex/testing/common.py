# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common utilities for testing."""
# pylint: disable=too-many-locals, unused-import, too-many-arguments, protected-access
import copy
import functools
import logging
import os
import random
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

import raf
from raf import distributed as dist
import ratex
from ratex.utils.cache import Cache, cache
from ..lazy_tensor_core.core import lazy_model as lm


class TorchLeNet(nn.Module):
    """LeNet in PyTorch."""

    # pylint: disable=missing-docstring
    def __init__(self, input_shape=28, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, bias=False)
        self.linear1 = nn.Linear(((input_shape // 2 - 4) // 2) ** 2 * 16, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.relu(out)
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = self.conv2(out)
        out = torch.relu(out)  # pylint: disable=no-member
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = torch.flatten(out, 1)  # pylint: disable=no-member
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out


def default_logger():
    """A logger used to output seed information to logs."""
    new_logger = logging.getLogger(__name__)
    # getLogger() lookups will return the same logger, but only add the handler once.
    if len(new_logger.handlers) == 0:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        new_logger.addHandler(handler)
        if new_logger.getEffectiveLevel() == logging.NOTSET:
            new_logger.setLevel(logging.INFO)
    return new_logger


logger = default_logger()  # pylint: disable=invalid-name


def with_seed(seed=None):
    """
    A decorator for test functions that manages rng seeds.

    Parameters
    ----------

    seed : the seed to pass to np.random and random


    This tests decorator sets the np and python random seeds identically
    prior to each test, then outputs those seeds if the test fails or
    if the test requires a fixed seed (as a reminder to make the test
    more robust against random data).

    @with_seed()
    def test_ok_with_random_data():
        ...

    @with_seed(1234)
    def test_not_ok_with_random_data():
        ...

    Use of the @with_seed() decorator for all tests creates
    tests isolation and reproducability of failures.  When a
    test fails, the decorator outputs the seed used.
    """

    def test_helper(orig_test):
        @functools.wraps(orig_test)
        def test_new(*args, **kwargs):
            if seed is not None:
                this_test_seed = seed
                log_level = logging.INFO
            else:
                this_test_seed = np.random.randint(0, np.iinfo(np.int32).max)
                log_level = logging.DEBUG
            post_test_state = np.random.get_state()
            np.random.seed(this_test_seed)
            random.seed(this_test_seed)
            # 'pytest --logging-level=DEBUG' shows this msg even with an ensuing core dump.
            pre_test_msg = (
                f"Setting test np/python random seeds, use seed={this_test_seed}" " to reproduce."
            )
            on_err_test_msg = (
                f"Error seen with seeded test, use seed={this_test_seed}" " to reproduce."
            )
            logger.log(log_level, pre_test_msg)
            try:
                ret = orig_test(*args, **kwargs)
            except:
                # With exceptions, repeat test_msg at WARNING level to be sure it's seen.
                if log_level < logging.WARNING:
                    logger.warning(on_err_test_msg)
                raise
            finally:
                # Provide test-isolation for any test having this decorator
                np.random.set_state(post_test_state)
            return ret

        return test_new

    return test_helper


def with_temp_cache(orig_test):
    """
    A decorator for test functions to use temporary cache folder.
    NOTES: This creates a temporary new global cache, which will affect other tests result
    if multi tests are being exectuted in parallel. Need to revise in the future if we run tests
    in parallel.
    """

    @functools.wraps(orig_test)
    def wrapper(*args, **kwargs):
        # Backup the original cache.
        persist_path = str(cache.persist_path)

        with TemporaryDirectory(prefix="ratex_test_") as temp_dir:
            # Hook persistent cache to a temporary one
            Cache.__init__(cache, temp_dir)
            ret = orig_test(*args, **kwargs)

        # Recover the cache.
        Cache.__init__(cache, persist_path)
        return ret

    return wrapper


def with_dumped_tensor_file(orig_test):
    """A decorator for test functions to dump tensor files for verification."""

    @functools.wraps(orig_test)
    def wrapper(*args, **kwargs):
        with TemporaryDirectory(prefix="ratex_test_") as temp_dir:
            with patch.dict(
                os.environ,
                {
                    "COMPILATION_CACHE_SIZE": "0",
                    "LTC_SAVE_TENSORS_FILE": str(Path(temp_dir) / "ltc_file.txt"),
                },
            ):
                return orig_test(*args, **kwargs)

    return wrapper


def dryrun_dumped_ir_file(orig_test):
    """Dry run and dump the IR file for verification."""

    @functools.wraps(orig_test)
    def wrapper(*args, **kwargs):
        with TemporaryDirectory(prefix="ratex_test_") as temp_dir:
            with patch.dict(
                os.environ,
                {
                    "RATEX_DRY_RUN": "true",
                    "RATEX_SAVE_IR_FILE": str(Path(temp_dir) / "module.json"),
                    "RATEX_DUMP_ALIAS": str(Path(temp_dir) / "alias.txt"),
                },
            ):
                return orig_test(*args, **kwargs)

    return wrapper


def with_enable_param_aliasing(orig_test, enable=True):
    """A decorator for test functions to dump tensor files for verification."""

    @functools.wraps(orig_test)
    def wrapper(*args, **kwargs):

        with patch.dict(
            os.environ,
            {
                "ENABLE_PARAM_ALIASING": ("true" if enable else "false"),
            },
        ):
            return orig_test(*args, **kwargs)

    return wrapper


def with_mock_distributed_info(world_size, rank, zero_opt_level=0, enable_data_parallel=False):
    """
    A decorator for testing fucntions with mock distributed context. This also sets the CPP values.
    """

    def test_helper(orig_test):
        @functools.wraps(orig_test)
        def wrapper(*args, **kwargs):
            dist.set_default_communicator("void")
            dcfg = dist.get_config()
            comm = dist.get_communicator()
            old_dcfg = dcfg.dumps()
            old_comm = comm.dumps()
            comm.size = world_size
            comm.rank = rank
            dcfg.zero_opt_level = zero_opt_level
            dcfg.enable_data_parallel = enable_data_parallel

            retval = orig_test(*args, **kwargs)

            dcfg.loads(old_dcfg)
            comm.loads(old_comm)
            return retval

        return wrapper

    return test_helper


def fake_image_dataset(batch, channel, image_size, num_classes, dtype=torch.float32):
    """Fake an image dataset."""
    from torchvision import datasets, transforms  # pylint: disable=import-outside-toplevel

    return datasets.FakeData(
        size=batch,
        image_size=(channel, image_size, image_size),
        num_classes=num_classes,
        transform=transforms.Compose(
            [
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype),
            ]
        ),
    )


def train(
    device,
    model,
    dataset,
    criterion=lambda pred, true: -torch.sum(pred * true) / true.size(0),
    optimizer=optim.SGD,
    optimizer_params=None,
    batch_size=1,
    num_classes=10,
    num_epochs=3,
    amp=False,
    trim=False,
    reduce_gradients=False,
    dtype=torch.float32,
    epilogue_closure=None,
    set_to_none=True,
    acc_grad_steps=None,
    force_one_hot=True,
    return_model=False,
):
    """Run training."""
    if optimizer_params is None:
        optimizer_params = {"lr": 0.001}

    results = []
    model = copy.deepcopy(model)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    dataset_size = len(dataset)
    model.train()

    if "lazy" in device:
        model = ratex.jit.script(model)
    model = model.to(device, dtype=dtype)
    optimizer = optimizer(model.parameters(), **optimizer_params)

    for epoch in range(num_epochs):
        logger.debug("Epoch %2d starts...", epoch)
        start = time.time()
        running_loss = []
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            if force_one_hot:
                labels = torch.from_numpy(np.eye(num_classes, dtype=np.float32)[labels]).to(dtype)
            labels = labels.to(device)
            with ratex.amp.autocast(amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / acc_grad_steps if acc_grad_steps else loss
                loss.backward()
                if trim:
                    logger.debug("Mark Step...")
                    lm.mark_step()
                if not acc_grad_steps or idx % acc_grad_steps == (acc_grad_steps - 1):
                    if reduce_gradients:
                        ratex.core.lazy_model.reduce_gradients(optimizer)
                    optimizer.step()
                    logger.debug("Mark Step...")
                    if set_to_none:
                        optimizer.zero_grad(set_to_none=set_to_none)
                    else:
                        optimizer.zero_grad(inplace_update=True)
                    lm.mark_step()
                    running_loss.append((loss, inputs.size(0)))

            if epilogue_closure:
                epilogue_closure()

        epoch_loss = sum([l.item() * w for l, w in running_loss]) / dataset_size
        end = time.time()
        logger.debug("Epoch %2d, Loss %.4f, Time %.4f", epoch, epoch_loss, (end - start))
        results.append(epoch_loss)
    return results if not return_model else (results, model)


def verify(lazy_results, cpu_results, tol=1e-5):
    """Verify the series of losses."""
    logger.debug("lazy_losses = %s", lazy_results)
    logger.debug("cpu_losses = %s", cpu_results)
    for lazy, cpu in zip(lazy_results, cpu_results):
        torch.testing.assert_close(lazy, cpu, atol=tol, rtol=tol)


def run_step(device, model_origin, args, jit_script=True, with_backward=False):
    """
    Run the model once.

    Parameters
    ----------
    device : str
        Device to run on

    model_origin : torch.nn.Module
        The original PyTorch model

    args : List[torch.Tensor]
        Args of the model

    jit_script :
        If False, the graph will be traced directly instead of leveraging TorchScript
        and AutoDiff. This is used to evaluate lowering the ops without backward.
    """
    model = copy.deepcopy(model_origin)
    if device == "lazy" and jit_script:
        model = ratex.jit.script(model)
    model = model.to(device, dtype=torch.float32)
    args = [arg.to(device) for arg in args]
    out = model(*args)

    if with_backward:
        loss = out.sum()
        loss.backward()

    lm.mark_step()
    if isinstance(out, tuple):
        out = [o.to("cpu") for o in out]
    else:
        out = out.to("cpu")

    if with_backward:
        grads = [a.grad.to("cpu") for a in args]
        return (out, grads)

    return out


def verify_step(model, args, jit_script=True, with_backward=False, tol=1e-5):
    """Verify the results between CPU and Lazy"""
    if with_backward:
        args_ratex = []
        args_cpu = args
        for x in args:
            n_x = np.copy(x.data)
            t_x_ratex = torch.tensor(n_x, device="lazy", dtype=torch.float32, requires_grad=True)
            args_ratex.append(t_x_ratex)
        out_cpu, grad_cpu = run_step("cpu", model, args_cpu, with_backward=with_backward)
        out_lazy, grad_lazy = run_step("lazy", model, args_ratex, jit_script, with_backward)
        torch.testing.assert_close(grad_cpu, grad_lazy, rtol=tol, atol=tol)
        torch.testing.assert_close(out_cpu, out_lazy, rtol=tol, atol=tol)
    else:
        torch.testing.assert_close(
            run_step("cpu", model, args, with_backward=with_backward),
            run_step("lazy", model, args, jit_script=jit_script, with_backward=with_backward),
            rtol=tol,
            atol=tol,
        )


def compile_model(model_origin, args, jit_script=True):
    """
    Trace and compile the model without execution.

    Parameters
    ----------
    model_origin : torch.Module
      The original PyTorch model

    args : List[torch.Tensor]
      Input args of the model

    jit_script : bool
      If False, the graph will be traced using LTC-to-raf lowering
      instead of raf.frontend.from_pytorch in jit.script, and it is used to evaluate lowering the
      ops without backward.

    Return
    ------
    module : raf.Module
      Compiled raf module
    """

    def _compile(model_origin, args, jit_script):
        meta_ir_file = os.environ["RATEX_SAVE_IR_FILE"]
        run_step("lazy", model_origin, args, jit_script)

        with open(meta_ir_file, "r") as module_file:
            module_json = module_file.read()
            module = raf.ir.serialization.LoadJSON(module_json)
        return module

    return dryrun_dumped_ir_file(_compile)(model_origin, args, jit_script)


def numpy(x):
    """Helper function to convert x to numpy"""
    if isinstance(x, (raf.ndarray, raf._core.value.TensorValue)):
        return x.numpy()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if np.isscalar(x):
        return np.array(x)
    assert isinstance(x, np.ndarray), f"{type(x)} is not supported"
    return x


def check(m_x, m_y, *, rtol=1e-5, atol=1e-5):
    """Helper function to check if m_x and m_y are equal"""
    m_x = numpy(m_x)
    m_y = numpy(m_y)
    np.testing.assert_allclose(m_x, m_y, rtol=rtol, atol=atol)


def get_most_recent_alias():
    """Return the input to output alias of the most recent compiled module"""
    alias_dump_file = os.environ["RATEX_DUMP_ALIAS"]
    assert alias_dump_file
    alias = {}
    with open(alias_dump_file, "r") as alias_file:
        alias_text = alias_file.read()
        alias = {line.split()[0]: line.split()[1] for line in alias_text.splitlines()}
    return alias
