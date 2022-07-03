from functools import wraps
import torch

class TimeMeasure:
    time_result = {}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    @classmethod
    def stop_watch(cls, func):
        time_result = cls.time_result

        @wraps(func)
        def wrapper(*args, **kargs):
            cls.start.record()
            result = func(*args, **kargs)
            cls.end.record()
            torch.cuda.synchronize()
            elapsed_time = cls.start.elapsed_time(cls.end)

            if not func.__name__ in time_result:
                time_result[func.__name__] = []
            time_result[func.__name__].append(elapsed_time)

            return result

        return wrapper

    @classmethod
    def show_time_result(cls, main_func, sub_funcs=None):
        time_result = cls.time_result

        main_func_name = main_func.__name__
        time_func = sum(time_result[main_func_name])
        print(f"{main_func_name}: {time_func:.3f}[msec.]")

        if sub_funcs is None:
            sub_func_names = set(time_result.keys()) - set([main_func.__name__])
        else:
            sub_func_names = list(map(lambda func: func.__name__, sub_funcs))

        max_len_names = max(map(lambda name: len(name), sub_func_names))

        for sub_func_name in sub_func_names:
            time_func = sum(time_result[sub_func_name])
            format_str = "  {:<" + str(max_len_names) + "}: {:.3f}[msec.]"
            print(format_str.format(sub_func_name, time_func) )

    @classmethod
    def reset_time_result(cls):
        time_result = cls.time_result
        time_result.clear()