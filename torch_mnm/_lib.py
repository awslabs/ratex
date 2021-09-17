import os, sys
torch_mnm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
meta_python_path = os.path.join(torch_mnm_path, "third_party/meta/python")
tvm_python_path = os.path.join(torch_mnm_path, "third_party/meta/3rdparty/tvm/python")
os.environ["TVM_LIBRARY_PATH"] = str(os.path.join(torch_mnm_path, "torch_mnm/lib"))
sys.path.insert(1, meta_python_path)
sys.path.insert(1, tvm_python_path)

import mnm
import tvm

del sys.path[1:3]
os.environ.pop("TVM_LIBRARY_PATH")
