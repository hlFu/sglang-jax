import os

SGLANG_DEBUG = os.environ.get("SGLANG_DEBUG", "0") == "1"

def log_shardings(name):
    def decorator(fn):
        if not SGLANG_DEBUG:
            return fn
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for i, a in enumerate(args):
                if hasattr(a, 'aval') and hasattr(a.aval, 'sharding'):
                    print(f"{name} input[{i}]: {a.aval.shape} {a.aval.sharding}")
            result = fn(*args, **kwargs)
            if hasattr(result, 'aval') and hasattr(result.aval, 'sharding'):
                print(f"{name} output: {result.aval.shape} {result.aval.sharding}")
            elif isinstance(result, tuple):
                for i, r in enumerate(result):
                    if hasattr(r, 'aval') and hasattr(r.aval, 'sharding'):
                        print(f"{name} output[{i}]: {r.aval.shape} {r.aval.sharding}")
            return result
        return wrapper
    return decorator