import sys
import platform
import psutil

# 这个模块提供了调试工具，用于打印系统信息和资源使用情况

DIVIDER = "[---------] 系统信息 [---------]"

def print_system_info():
    """打印系统信息和资源使用情况
    
    输出包括Python版本、平台信息、CPU信息、内存信息和磁盘信息。
    所有输出都格式化为易于阅读的形式。
    """
    print(DIVIDER)
    print()
    print("Python版本:")
    print(f"  {sys.version}")

    print("\n平台信息:")
    print(f"  系统: {platform.system()}")
    print(f"  发行版: {platform.release()}")
    print(f"  版本: {platform.version()}")
    print(f"  机器类型: {platform.machine()}")
    print(f"  处理器: {platform.processor()}")

    print("\nCPU信息:")
    print(f"  物理核心数: {psutil.cpu_count(logical=False)}")
    print(f"  总核心数: {psutil.cpu_count(logical=True)}")
    cpu_freq = psutil.cpu_freq()
    print(f"  最大频率: {cpu_freq.max:.2f} Mhz")
    print(f"  当前频率: {cpu_freq.current:.2f} Mhz")

    print("\n内存信息:")
    vm = psutil.virtual_memory()
    print(f"  总计: {vm.total / (1024**3):.2f} GB")
    print(f"  可用: {vm.available / (1024**3):.2f} GB")
    print(f"  已用: {vm.used / (1024**3):.2f} GB")

    print("\n磁盘信息:")
    partitions = psutil.disk_partitions()
    for partition in partitions:
        print(f"  设备: {partition.device}")
        print(f"    挂载点: {partition.mountpoint}")
        try:
            disk_usage = psutil.disk_usage(partition.mountpoint)
            print(f"      总大小: {disk_usage.total / (1024**3):.2f} GB")
            print(f"      已用: {disk_usage.used / (1024**3):.2f} GB")
            print(f"      可用: {disk_usage.free / (1024**3):.2f} GB")
        except PermissionError:
            print("      权限被拒绝")

    print()
    print(DIVIDER)
