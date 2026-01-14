


import subprocess
import pandas as pd
from datetime import datetime
import os

def collect_slurm_data():
    os.environ["SLURM_CONF"] = "/etc/slurm/slurm.conf"
    start_date = (datetime.now() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    # 命令格式
    cmd = f'sacct --starttime={start_date} --endtime={end_date} --format=JobID,User,Partition,AllocCPUS,Elapsed,State -P'
    try:
        result = subprocess.Popen(
            f'echo "19970116Dzq@dzq" | sudo -S {cmd}',
            shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = result.communicate()

        if result.returncode != 0:
            raise Exception(stderr)
        
        # 将 stdout 解析为 DataFrame
        data = [line.split("|") for line in stdout.strip().split("\n")[1:]]
        df = pd.DataFrame(data, columns=["JobID", "User", "Partition", "AllocCPUS", "Elapsed", "State"])
        
        # 保存为 CSV 文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"/data/researchHome/pdou/hpc_cluster_report/reports/slurm_data_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"Slurm data collected and saved to {csv_path}.")
        return csv_path
    except Exception as e:
        print(f"Error collecting data: {e}")
        return None

if __name__ == "__main__":
    collect_slurm_data()
# pdsh -w compute[1-16] ps -ef   节点上检查直接运行的进程
# pdsh -w compute[1-2,4-16] ps -ef
# pdsh -w compute[1-2,4-16] "ps -ef | grep -v '^root'"