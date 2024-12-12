import time
import psutil
from collections import defaultdict
from pynvml import *

def get_gpu_processes():
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    processes = []

    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        try:
            procs = nvmlDeviceGetComputeRunningProcesses(handle)
            for proc in procs:
                try:
                    process = psutil.Process(proc.pid)
                    process_name = process.name()
                except psutil.NoSuchProcess:
                    process_name = "Unknown"
                processes.append({
                    'pid': proc.pid,
                    'name': process_name,
                    'gpu_util': nvmlDeviceGetUtilizationRates(handle).gpu
                })
        except NVMLError as err:
            print(f"Error: {err}")

    nvmlShutdown()
    return processes

def main():
    interval = 1  # 每秒采样一次
    total_time = 10  # 总采样时间为10秒
    samples = total_time // interval

    process_usage = defaultdict(lambda: defaultdict(list))

    for _ in range(samples):
        processes = get_gpu_processes()
        for process in processes:
            pid = process['pid']
            name = process['name']
            usage = process['gpu_util']
            process_usage[pid]['name'] = name
            process_usage[pid]['usage'].append(usage)
        time.sleep(interval)

    print("每个进程在统计时间段内的 GPU 占用率：")
    for pid, info in process_usage.items():
        name = info['name']
        usages = info['usage']
        average_usage = sum(usages) / len(usages)
        print(f"进程 {pid} ({name}): 平均 GPU 使用率: {average_usage:.2f}%")

if __name__ == "__main__":
    main()
