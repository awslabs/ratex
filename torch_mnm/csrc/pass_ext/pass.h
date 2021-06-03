/*!
 * Copyright (c) 2021 by Contributors
 * \file pass.h
 * \brief Customized MNM passes
 */
#pragma once

#include "mnm/pass.h"

namespace mnm {
namespace pass {

/*!
 * \brief Eliminate the closure in returned value. Replace them with constant 1s.
 * \return The pass.
 */
Pass EliminateClosure();


/*!
 * \brief Wrap identity output values with mnm.op.copy
 * \return The pass.
 */
Pass WrapIdentity();

}  // namespace pass
}  // namespace mnm
