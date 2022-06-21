# Copyright (c) 2018 Google Inc. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import ratex.lazy_tensor_core.core.lazy_model as ltm


class GradScaler(torch.cuda.amp.GradScaler):
    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        retval = None
        ltm.mark_step()
        if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
            retval = optimizer.step(*args, **kwargs)
        return retval
