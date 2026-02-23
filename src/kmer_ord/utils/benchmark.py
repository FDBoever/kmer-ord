# src/kmer_ord/utils/benchmark.py
import psutil
import csv
import os
import time
from datetime import datetime

class BenchmarkTimer:
    """
    Context manager for timing, CPU, and memory usage of operations.
    Automatically logs metrics to TSV.
    """
    def __init__(self, label="Run", log_dir="benchmarking", script_name=None,
                 input_file=None, input_args=None):
        self.label = label
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "benchmark_log.tsv")
        self.script_name = script_name
        self.input_file = str(input_file) if input_file else None
        self.input_args = input_args

        os.makedirs(self.log_dir, exist_ok=True)

        self.start_time = None
        self.start_cpu_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_cpu_time = time.process_time()
        self.start_memory = psutil.Process().memory_info().rss
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_cpu_time = time.process_time()
        end_memory = psutil.Process().memory_info().rss

        self.wall_time = end_time - self.start_time
        self.cpu_time = end_cpu_time - self.start_cpu_time
        self.memory_used = max(0, end_memory - self.start_memory)

        self._log_metrics()
    
    def _log_metrics(self):
        log_exists = os.path.isfile(self.log_file)
        input_file_size = os.path.getsize(self.input_file) if self.input_file and os.path.exists(self.input_file) else "N/A"

        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file, delimiter='\t')
            if not log_exists:
                writer.writerow([
                    "timestamp", "script_name", "input_file", "input_file_size_bytes",
                    "input_args", "label", "wall_time_s", "cpu_time_s", "memory_used_bytes"
                ])
            writer.writerow([
                datetime.now().isoformat(),
                self.script_name or "N/A",
                self.input_file or "N/A",
                input_file_size,
                self.input_args or "N/A",
                self.label,
                self.wall_time,
                self.cpu_time,
                self.memory_used
            ])