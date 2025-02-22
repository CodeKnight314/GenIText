import time
from dataclasses import dataclass
from functools import wraps

@dataclass
class TimingStats:
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0

class PerformanceTracker:
    def __init__(self): 
        self.functional_timings = {}
        self.subroutine_timings = {}
        
    def track_function(self, funcs): 
        @wraps(funcs)
        def wrapper(*args, **kwargs): 
            start = time.perf_counter()
            result = funcs(*args, **kwargs)
            end = time.perf_counter()
            execution_time = end - start
            self.functional_timings[funcs.__name__] = self.functional_timings.get(funcs.__name__, TimingStats())
            self.functional_timings[funcs.__name__].total_time += execution_time
            self.functional_timings[funcs.__name__].call_count += 1
            self.functional_timings[funcs.__name__].min_time = min(self.functional_timings[funcs.__name__].min_time, execution_time)
            self.functional_timings[funcs.__name__].max_time = max(self.functional_timings[funcs.__name__].max_time, execution_time)
            return result
        return wrapper
    
    def track_subroutine(self, snippets):
        class SnippetTimer: 
            def __init__(self, tracker): 
                self.tracker = tracker
                self.snippets = snippets 
            
            def __enter__(self):
                self.start = time.perf_counter()
                
            def __exit__(self, exc_type, exc_value, traceback):
                end = time.perf_counter()
                execution_time = end - self.start
                if self.snippets not in self.tracker.subroutine_timings: 
                    self.tracker.subroutine_timings[self.snippets] = TimingStats()
                self.tracker.subroutine_timings[self.snippets].total_time += execution_time
                self.tracker.subroutine_timings[self.snippets].call_count += 1
                self.tracker.subroutine_timings[self.snippets].min_time = min(self.tracker.subroutine_timings[self.snippets].min_time, execution_time)
                self.tracker.subroutine_timings[self.snippets].max_time = max(self.tracker.subroutine_timings[self.snippets].max_time, execution_time)
        
        return SnippetTimer(self)
    
    def print_stats(self): 
        print("Functional Timings:")
        for func_name, stats in self.functional_timings.items(): 
            print(f"Function: {func_name}")
            print(f"Total Time: {stats.total_time}")
            print(f"Call Count: {stats.call_count}")
            print(f"Min Time: {stats.min_time}")
            print(f"Max Time: {stats.max_time}")
            
        print("Subroutine Timings:")
        for snippet, stats in self.subroutine_timings.items(): 
            print(f"Snippet: {snippet}")
            print(f"Total Time: {stats.total_time}")
            print(f"Call Count: {stats.call_count}")
            print(f"Min Time: {stats.min_time}")
            print(f"Max Time: {stats.max_time}")
            
    def dump_info(self, filename):
        with open(filename, "w") as f: 
            f.write("Functional Timings:\n")
            for func_name, stats in self.functional_timings.items(): 
                f.write(f"Function: {func_name}\n")
                f.write(f"Total Time: {stats.total_time}\n")
                f.write(f"Call Count: {stats.call_count}\n")
                f.write(f"Min Time: {stats.min_time}\n")
                f.write(f"Max Time: {stats.max_time}\n")
                
            f.write("Subroutine Timings:\n")
            for snippet, stats in self.subroutine_timings.items(): 
                f.write(f"Snippet: {snippet}\n")
                f.write(f"Total Time: {stats.total_time}\n")
                f.write(f"Call Count: {stats.call_count}\n")
                f.write(f"Min Time: {stats.min_time}\n")
                f.write(f"Max Time: {stats.max_time}\n")
