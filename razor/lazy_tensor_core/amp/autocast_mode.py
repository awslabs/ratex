# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright (c) Facebook, Inc.

import torch

autocast = torch.cuda.amp.autocast
custom_fwd = torch.cuda.amp.custom_fwd
custom_bwd = torch.cuda.amp.custom_bwd
