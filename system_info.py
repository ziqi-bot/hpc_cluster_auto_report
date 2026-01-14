import re
import subprocess



def get_hostname():
    """Retrieve the hostname of the login node."""
    try:
        return subprocess.check_output("hostname", shell=True, text=True).strip()
    except Exception as e:
        print(f"Error retrieving hostname: {e}")
        return "Unknown"

def get_cpu_info():
    """Retrieve the core count and CPU model of the login node."""
    try:
        cpu_info = subprocess.check_output("lscpu", shell=True, text=True)
        cores = int(subprocess.check_output("nproc", shell=True, text=True).strip())
        model = "Unknown"
        for line in cpu_info.split("\n"):
            if "Model name" in line:
                model = line.split(":")[1].strip()
                break
        return cores, model
    except Exception as e:
        print(f"Error retrieving CPU info: {e}")
        return "Unknown", "Unknown"

def get_memory_usage():
    """Retrieve the memory usage of the login node."""
    try:
        memory_info = subprocess.check_output("free -h", shell=True, text=True).strip()
        for line in memory_info.split("\n"):
            if line.startswith("Mem:"):
                mem_data = line.split()
                total_memory = mem_data[1]
                free_memory = mem_data[3]
                return f"{free_memory} / {total_memory}"
        return "Unknown"
    except Exception as e:
        print(f"Error retrieving memory usage: {e}")
        return "Unknown"

def get_compute_nodes_memory_usage():
    """
    Retrieve memory usage information from compute nodes using pdsh.

    Returns:
        dict: A sorted dictionary containing node names and their free memory.
    """
    memory_usage = {}
    try:
        node_info = subprocess.check_output(
            "pdsh -w compute[1-16] free -h", shell=True, text=True
        ).strip()

        for line in node_info.split("\n"):
            if "Mem:" in line:
                node_name = line.split(":")[0].strip()  # 获取节点名称
                mem_data = line.split()  # 拆分内存数据
                total_memory = mem_data[2]  # 总内存
                free_memory = mem_data[4]  # 空闲内存
                memory_usage[node_name] = f"{free_memory}"  # 保存内存信息

    except Exception as e:
        print(f"Error retrieving memory usage: {e}")
        return {}

    # 对字典进行按节点名称排序并返回
    sorted_memory_usage = dict(sorted(
        memory_usage.items(),
        key=lambda x: int(re.search(r'\d+', x[0]).group())
    ))
    return sorted_memory_usage

def get_os_info():
    """Retrieve operating system version information."""
    try:
        os_info = subprocess.check_output("cat /etc/os-release", shell=True, text=True).strip()
        for line in os_info.split("\n"):
            if line.startswith("PRETTY_NAME"):
                return line.split("=")[1].strip().replace('"', '')
        return "Unknown OS"
    except Exception as e:
        print(f"Error retrieving OS info: {e}")
        return "Unknown OS"

def get_disk_usage():
    """Retrieve the disk usage of the login node."""
    try:
        disk_info = subprocess.check_output("df -h --total | grep total", shell=True, text=True).strip()
        parts = disk_info.split()
        total = parts[1]
        used = parts[2]
        return f"{used} / {total}"
    except Exception as e:
        print(f"Error retrieving disk usage: {e}")
        return "Unknown"

def get_all_users():
    """Retrieve the total number of users in SLURM."""
    try:
        users = subprocess.check_output("sacctmgr list user -P | cut -d '|' -f 1", shell=True, text=True).strip().split("\n")
        return len(users)
    except Exception as e:
        print(f"Error retrieving SLURM user count: {e}")
        return "Unknown"
    
def get_all_modules_with_versions():
    """Retrieve all available modules with their versions and save all output to a file."""
    try:
        # Run 'module avail' in a bash shell
        modules_output = subprocess.check_output(
            ["bash", "-c", "module avail"],
            text=True,
            stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as e:
        # Capture error output if the command fails
        modules_output = e.output.strip()
    except Exception as e:
        # Handle unexpected exceptions
        modules_output = f"Unexpected error: {str(e)}"

    # Save all output (success or error) to a file
    out_path = "/data/researchHome/pdou/hpc_cluster_report/modules_with_versions.txt"
    try:
        with open(out_path, "w") as f:
            f.write(modules_output)

        print(f"Output saved to {out_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    # Retrieve various system information
    hostname = get_hostname()
    cores, cpu_model = get_cpu_info()
    memory_usage = get_memory_usage()
    disk_usage = get_disk_usage()
    total_users = get_all_users()
    os_info = get_os_info()
    compute_memory_usage = get_compute_nodes_memory_usage()

    modules=get_all_modules_with_versions()
    

    # Save all information to file
    output_path = "/data/researchHome/pdou/hpc_cluster_report/system_info.txt"

    try:
        with open(output_path, "w") as f:
            f.write(f"Hostname: {hostname}\n")
            f.write(f"Operating System: {os_info}\n")
            f.write(f"Login Node FreeMem: {memory_usage}\n")
            f.write("Compute Nodes Mem: FreeMem / 187Gi\n")

            # Format compute nodes' memory usage, 3 per line
            compute_nodes = list(compute_memory_usage.items())
            for i in range(0, len(compute_nodes), 3):
                row = compute_nodes[i:i+3]
                
                f.write("Node List: "+" ".join([f"{node}: {free_mem}   " for node, free_mem in row]) + "\n")

            f.write(f"Disk Usage: {disk_usage}\n")
            f.write(f"Total Users: {total_users}\n")
            f.write(f"Login Node: {cpu_model} ({cores} cores)\n")
        print(f"System info saved to {output_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")

# pdsh -w compute[1-16] ps -ef   节点上检查直接运行的进程
# pdsh -w compute[1-2,4-16] ps -ef
# pdsh -w compute[1-2,4-16] "ps -ef | grep -v '^root'"

