from collections.abc import MutableMapping


def nested_fn(fn, obj):
    if isinstance(obj, MutableMapping):
        res = type(obj)()
        for key, value in obj.items():
            res[key] = nested_fn(fn, value)
    else:
        res = fn(obj)
    return res
