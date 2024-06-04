import cProfile
import functools

def profile_function(filename=None):
    def profiler_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            
            # Execute the function
            result = func(*args, **kwargs)
            
            profiler.disable()
            # Save stats to a file
            profiler.dump_stats(filename if filename else func.__name__ + '_profile.prof')
            
            return result
        return wrapper
    return profiler_decorator