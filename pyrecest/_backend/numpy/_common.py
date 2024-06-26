import numpy as _np

from pyrecest._backend._dtype_utils import (
    _dyn_update_dtype,
    _modify_func_default_dtype,
    get_default_cdtype,
    get_default_dtype,
)

from .._shared_numpy._common import (
    _add_default_dtype_by_casting,
    _allow_complex_dtype,
    _box_binary_scalar,
    _box_unary_scalar,
    _cast_fout_to_input_dtype,
    _cast_out_from_dtype,
    _cast_out_to_input_dtype,
    _get_wider_dtype,
    _is_boolean,
    _is_iterable,
    as_dtype,
    atol,
    cast,
    convert_to_wider_dtype,
    is_array,
    is_bool,
    is_complex,
    is_floating,
    rtol,
    set_default_dtype,
    to_ndarray,
)

array = _cast_out_from_dtype(target=_np.array, dtype_pos=1)
eye = _modify_func_default_dtype(target=_np.eye)
zeros = _dyn_update_dtype(target=_np.zeros, dtype_pos=1)
