import time
import yaml
import functools
import numpy as np

def count_time(print_args_index=None):
    """
    Count the execution time of a function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            if print_args_index is not None:
                tag = args[print_args_index]
                print(f"{tag} K done in {elapsed:.2f} s.\n")
            else:
                print(f"{func.__name__} done in {elapsed:.2f} s.")

            return result
        return wrapper
    return decorator

def pad_ragged_2d(data, *, fill_value, dtype,) -> np.ndarray:
    """Pad a ragged 2D list to a 2D numpy array."""
    n = len(data)
    m = max(len(row) for row in data)

    out = np.full((n, m), fill_value, dtype=dtype)
    for i, row in enumerate(data):
        out[i, :len(row)] = row

    return out