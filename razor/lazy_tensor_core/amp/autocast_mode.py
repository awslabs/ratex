# Copyright (c) 2018 Google Inc. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

autocast = torch.cuda.amp.autocast
custom_fwd = torch.cuda.amp.custom_fwd
custom_bwd = torch.cuda.amp.custom_bwd
