"""Value registry."""
from mnm._lib import _APIS, _get_apis

# Reload APIs to ensure functions from torch_mnm are included
_APIS = _get_apis()

ValueToHandle = _APIS.get("mnm.value.ValueToHandle", None)
HandleToValue = _APIS.get("mnm.value.HandleToValue", None)
