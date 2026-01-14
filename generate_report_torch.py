
import pandas as pd # type: ignore
import matplotlib.pyplot as plt# type: ignore
from reportlab.lib.pagesizes import letter# type: ignore
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle# type: ignore
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle# type: ignore
from reportlab.lib.utils import ImageReader# type: ignore
from reportlab.lib import colors # type: ignore
from datetime import datetime
import glob
import os
import subprocess
import re
import torch# type: ignore
import torch.distributed as dist
import time

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from PIL import Image as PILImage  # 确保有这行



def read_module_info():
    """从文件读取登录节点的系统信息，并提取所有模块和版本信息。"""
    system_info_path = "/data/researchHome/pdou/hpc_cluster_report/modules_with_versions.txt"
    try:
        with open(system_info_path, "r") as f:
            lines = f.readlines()

        modules_with_versions = []
        for line in lines:
            # 跳过空行和分隔符
            if line.strip() == "" or line.startswith("-") or line.startswith("  Where:"):
                continue

            # 提取模块信息：按空格分割每行，处理多列模块
            columns = re.split(r" {2,}", line.strip())  # 根据两个或更多空格分割
            for col in columns:
                match = re.match(r"^([a-zA-Z0-9_./-]+)(?:\s+\((D|L)\))?$", col)
                if match:
                    module_name = match.group(1)
                    modules_with_versions.append(module_name)

        return modules_with_versions
    except FileNotFoundError:
        print(f"Error: File not found at {system_info_path}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def read_system_info():
    """从文件读取登录节点的系统信息，并处理多个 Compute Nodes 信息"""
    system_info_path = "/data/researchHome/pdou/hpc_cluster_report/system_info.txt"
    try:
        with open(system_info_path, "r") as f:
            lines = f.readlines()
        
        cluster_info = {}
        for line in lines:
            key, value = line.strip().split(":", 1)
            key = key.strip()
            value = value.strip()

            if key == "Node List":  # 如果是 Compute Node 信息
                if key not in cluster_info:
                    cluster_info[key] = []  # 初始化为列表
                cluster_info[key].append(value)  # 追加节点信息
            else:
                cluster_info[key] = value  # 其他键直接存储

        # 将 Compute Nodes 列表合并成字符串
        if "Node List" in cluster_info:
            cluster_info["Node List"] = "\n".join(cluster_info["Node List"])

        return cluster_info

    except Exception as e:
        print(f"Error reading system info: {e}")
        return {}


def summarize_compute_nodes():
    """汇总计算节点的状态、核心分布以及硬件信息"""
    try:
        # 使用 lscpu 获取计算节点的 CPU 型号
        lscpu_output = subprocess.check_output("lscpu", shell=True, text=True)
        model_name = parse_lscpu_output(lscpu_output)
        cores = int(subprocess.check_output("nproc", shell=True, text=True).strip())

        # 使用 sinfo 获取所有节点信息
        nodes_info = subprocess.check_output("sinfo -N -h", shell=True, text=True).strip().split("\n")
        if not nodes_info:
            return "No compute nodes detected"

        # 初始化存储
        idle_nodes = 0
        allocated_nodes = 0
        down_nodes = 0

        for node in nodes_info:
            parts = node.split()
            status = parts[-1]  # 节点状态
            if "idle" in status:
                idle_nodes += 1
            elif "alloc" in status or "mix" in status:
                allocated_nodes += 1
            elif "down" in status:
                down_nodes += 1

        # 统计总节点数
        total_nodes = len(nodes_info)

        # 构造汇总信息
        summary = (
            # f"{model_name} ({cores} cores per node)\n"
            f"{model_name} (36 cores per node)\n"
            f"{total_nodes} nodes "
            f"({idle_nodes} idle, {allocated_nodes} allocated, {down_nodes} down)"            
        )
        return summary

    except Exception as e:
        print(f"Error retrieving compute node information: {e}")
        return "Error retrieving compute node information"


def parse_lscpu_output(output):
    """解析 lscpu 的输出以提取 CPU 型号"""
    for line in output.split("\n"):
        if "Model name" in line:
            return line.split(":")[1].strip()
    return "Unknown CPU"

def get_cluster_info():
    """合并登录节点信息和计算节点信息"""
    login_info = read_system_info()
    compute_node_summary = summarize_compute_nodes()
    login_info["Compute Nodes"] = compute_node_summary
    return login_info



def get_latest_csv():
    """获取最新生成的 CSV 文件路径"""
    csv_files = glob.glob("/data/researchHome/pdou/hpc_cluster_report/reports/slurm_data_*.csv")
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the reports directory.")
    latest_file = max(csv_files, key=os.path.getctime)  # 获取最新生成的文件
    return latest_file




def generate_summary(df,cluster_info,dept_summary):
    # Model and pipeline configuration
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    text_gen_pipeline = pipeline(
        "text-generation",
        model=MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Data statistics
    total_jobs = len(df)
    users = df["User"].nunique()
    completed_jobs = len(df[df["State"] == "COMPLETED"])
    failed_jobs = len(df[df["State"] == "FAILED"])

    # Refined prompt, Prompt Engineering
    input_text = (
        "Generate a formal analytical summary based on the provided usage data." 
        "The summary should be approximately 400 words, structured with clear headings for better readability."
        "Focus on emphasizing the positive aspects of the data, written in past tense, and include a forward-looking conclusion."
        "Only include the summary text meant for the audience, without explicitly referencing the prompt."
        "The UNBC HPC cluster serves as a critical computational resource."
        "Cluster Usage Data (Past Year)\n"
        f"- Slurm users: {users}\n"
        f"- Job Distribution by Department: {dept_summary}\n"
        f"- Total jobs: {total_jobs}\n"
        f"- Completed jobs: {completed_jobs}\n"
        f"- Failed jobs: {failed_jobs}\n\n"
    )   

    # Generate text
    outputs = text_gen_pipeline(
        input_text,
        max_new_tokens=512,  # Limit output length
        min_new_tokens=256,
        temperature=0.3,  # Reduced randomness
        pad_token_id=128001,  # default 
        repetition_penalty=1.1,
        num_return_sequences=1  # Single output
    )
    generated_text = outputs[0]["generated_text"]

    # Post-process: Remove input text from the output
    summary_text = generated_text[len(input_text):].strip() if generated_text.startswith(input_text) else generated_text



    return summary_text



def analyze_data(csv_path):
    """分析数据并生成图表"""
    df = pd.read_csv(csv_path)
    figures_dir = "/data/researchHome/pdou/hpc_cluster_report/figures/"
   
    
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    try:
        

        # 用户任务柱状图
        user_summary = df["User"].value_counts().head(10)
        plt.figure(figsize=(10, 6))
        bars = user_summary.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title("Job Counts by User (Last Year)", fontsize=16)
        plt.xlabel("Users", fontsize=14)
        plt.ylabel("Job Count", fontsize=14)
        plt.xticks(rotation=45, fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # 添加数值标签
        for bar in bars.patches:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{int(bar.get_height())}",
                ha="center",
                fontsize=14,
            )

        plt.tight_layout()
        plt.savefig(f"{figures_dir}/user_job_counts.png", dpi=300)
        plt.close()



#####################################################################
        # User to department mapping
        user_dept_mapping = {
            "bernier": "Physics",
            "mandy": "Chemistry",
            "pdou": "Computer Science",
            "guoj2": "Environmental Science",
            "weththasi": "Environmental Science",
            "pghafarian": "Environmental Science",
            "taghizade": "Environmental Science",
            "aravindak": "Chemistry",
            "kafle": "Environmental Science"
        }

        # Filter and map users to departments
        df_top_users = df[df["User"].isin(user_dept_mapping.keys())].copy()
        df_top_users["Department"] = df_top_users["User"].map(user_dept_mapping)

        # Compute department usage summary
        dept_summary = df_top_users["Department"].value_counts()
        dept_percent = (dept_summary / dept_summary.sum() * 100).round(1)
        dept_usage_summary = {dept: f"{pct}%" for dept, pct in dept_percent.items()}

        colors = ["gold", "skyblue", "lightgreen", "salmon"]

        # Plot department job distribution
        plt.figure(figsize=(8, 8))
        wedges, _, autotexts = plt.pie(
            dept_summary,
            labels=["" for _ in dept_summary],
            autopct='%1.1f%%',
            startangle=140,
            colors=colors[:len(dept_summary)],
            textprops={'fontsize': 12},
        )
        plt.legend(wedges, dept_summary.index, title="Department", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.title("Job Distribution by Department (Top Users)", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/department_job_distribution.png", dpi=300)
        plt.close()

        # Job state mapping
        state_summary = df["State"].replace(
            {x: "CANCELLED" for x in df["State"].unique() if "CANCELLED" in x}
        ).replace({"RUNNING": "ACTIVE", "PENDING": "ACTIVE", "TIMEOUT": "FAILED"}).value_counts()

        # Ensure consistent formatting for the second pie chart but include number counts
        plt.figure(figsize=(8, 8))
        wedges, _, autotexts = plt.pie(
            state_summary,
            labels=[f"{key} ({value})" for key, value in state_summary.items()],
            autopct='%1.1f%%',
            startangle=140,
            colors=colors[:len(state_summary)],
            textprops={'fontsize': 12},
        )
        plt.legend(wedges, state_summary.index, title="Job State", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        plt.title("Job States Distribution", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/job_states.png", dpi=300)
        plt.close()


    #   ######################################
        

        # CPU 分配箱线图
        df["AllocCPUS"] = pd.to_numeric(df["AllocCPUS"], errors="coerce")
        plt.figure(figsize=(8, 6))
        boxplot = df["AllocCPUS"].plot(kind="box", patch_artist=True, notch=True)
        plt.title("CPU Allocation Summary", fontsize=16)
        plt.ylabel("Number of CPUs", fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # 添加统计摘要
        stats = df["AllocCPUS"].describe()
        textstr = "\n".join(
            [f"{key}: {value:.2f}" for key, value in stats.items() if key != "count"]
        )
        plt.gcf().text(0.15, 0.7, f"Summary Stats:\n{textstr}", fontsize=14, bbox=dict(facecolor="white", alpha=0.5))

       
        

        plt.tight_layout()
        plt.savefig(f"{figures_dir}/cpu_allocation.png", dpi=300)
        plt.close()





    except Exception as e:
        print(f"Error during chart generation: {e}")
        raise

    return df , dept_usage_summary


def header(canvas, doc):
    """自定义页眉函数，添加半透明效果的图片"""
    canvas.saveState()

    # 页眉图片路径
    header_path = "/data/researchHome/pdou/hpc_cluster_report/figures/header.png"

    if os.path.exists(header_path):
        img_reader = ImageReader(header_path)
        
        # 自动根据图片调整宽高
        img_width, img_height = img_reader.getSize()
        scale_factor = 0.4  # 控制缩放比例，可以根据需要修改
        new_width = img_width * scale_factor
        new_height = img_height * scale_factor

        # 图片位置设置为居中
        # x_position = (doc.pagesize[0] - new_width) / 2  # 水平居中
        x_position = doc.leftMargin   # 
        y_position = doc.pagesize[1] - new_height - 20  # 距顶部 20pt

        # 绘制半透明背景框
        canvas.setFillColorRGB(1, 1, 1, alpha=0.5)  # 白色半透明背景
        canvas.rect(x_position, y_position, new_width, new_height, stroke=0, fill=1)

        # 绘制图片
        canvas.drawImage(img_reader, x_position, y_position, width=new_width, height=new_height)

    canvas.restoreState()





def create_pdf_report(cluster_info, df, summary_text, pdf_path,module_versions):
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(name='TitleStyle', parent=styles['Title'], fontSize=20, leading=24, spaceAfter=20)
    style_subtitle = ParagraphStyle(name='SubTitleStyle', parent=styles['Heading2'], fontSize=14, spaceAfter=10)
    style_body = styles['BodyText']
    style_body.spaceAfter = 10

    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=80, bottomMargin=50)
    story = []

    # 首页
    story.append(Paragraph("High-Performance Computing Cluster Usage Report", style_title))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Ziqi (Peter) Dou", style_body))
    story.append(Paragraph("HPC Senior Lab Instructor", style_body))
    story.append(Paragraph("University of Northern British Columbia", style_body))
    # story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style_body))
    story.append(PageBreak())

    # 硬件和系统信息页
    story.append(Paragraph("System and Hardware Overview", style_subtitle))
    story.append(Paragraph("The following table summarizes the cluster's hardware and operating environment:", style_body))

    data = [["Property", "Details"]]
    for key, value in cluster_info.items():
        data.append([key, str(value)])

    table = Table(data, colWidths=[120, 350])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME', (0,0),(-1,0), 'Helvetica-Bold'),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    story.append(PageBreak())

    # 模块信息页
    story.append(Paragraph("Module Information", style_subtitle))
    story.append(Paragraph("The following table lists the available modules and their versions:", style_body))

    module_data = [["Module", "Version"]]
    for module in module_versions:
        if "/" in module:
            module_name, module_version = module.split("/", 1)
            module_data.append([module_name, module_version])
        else:
        # 如果模块没有版本号，只记录模块名，版本留空
            module_data.append([module, "N/A"])


    module_table = Table(module_data, colWidths=[200, 200])
    module_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME', (0,0),(-1,0), 'Helvetica-Bold'),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ]))
    story.append(module_table)
    story.append(Spacer(1, 20))
    story.append(PageBreak())

    

    # 使用情况统计与图表
    story.append(Paragraph("Usage Statistics and Visualizations", style_subtitle))
    story.append(Paragraph("The charts below illustrate recent job distribution, resource allocation, and job states over the last year.", style_body))
    story.append(Spacer(1, 20))

    figures_dir = "/data/researchHome/pdou/hpc_cluster_report/figures/"
    charts = ["user_job_counts","department_job_distribution", "job_states", "cpu_allocation"]

    for chart in charts:
        chart_path = os.path.join(figures_dir, f"{chart}.png")
        if os.path.exists(chart_path):
            story.append(Paragraph(chart.replace("_", " ").title(), styles['Heading3']))
            im = PILImage.open(chart_path)
            orig_width, orig_height = im.size
            new_width = 450.0
            ratio = new_width / orig_width
            new_height = orig_height * ratio

            img = Image(chart_path, width=new_width, height=new_height)
            story.append(img)
            story.append(Spacer(1, 20))
            story.append(PageBreak())

    

    # 总结与分析页
    story.append(Paragraph("Analytical Summary", style_subtitle))
    formatted_summary = summary_text.replace("\n", "<br />")
    story.append(Paragraph(formatted_summary, style_body))

  



    # doc.build(story)
    doc.build(story, onFirstPage=header, onLaterPages=header)
    print(f"PDF report generated successfully at: {pdf_path}")


def generate_summary(df, cluster_info, dept_summary, audience="internal"):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device=-1  # CPU only
    )

    total_jobs = len(df)
    users = df["User"].nunique()
    completed = len(df[df["State"] == "COMPLETED"])
    failed = len(df[df["State"] == "FAILED"])

    if audience == "external":
        tone = (
        "Generate a concise, business-oriented summary (approximately 400 words) of the HPC cluster's performance over the past year. "
        "The tone should be professional, optimistic, and tailored for non-technical stakeholders such as university leadership or external partners. "
        "Focus on key achievements, utilization highlights, and forward-looking statements. "
        "Avoid technical jargon and unnecessary detail. Do not include the prompt in the output."
    )
    else:
        tone = (
            "Generate a formal analytical summary based on the provided usage data. "
            "The summary should be approximately 400 words, structured with clear headings for better readability. "
            "Focus on emphasizing the positive aspects of the data, written in past tense, and include a forward-looking conclusion. "
            "Only include the summary text meant for the audience, without explicitly referencing the prompt."
        )

    prompt = (
        f"{tone}\n"
        f"The UNBC HPC cluster serves as a critical computational resource.\n"
        f"Cluster Usage Data (Past Year)\n"
        f"- Slurm users: {users}\n"
        f"- Job Distribution by Department: {dept_summary}\n"
        f"- Total jobs: {total_jobs}\n"
        f"- Completed jobs: {completed}\n"
        f"- Failed jobs: {failed}\n"
    )

    result = pipe(
        prompt,
        max_new_tokens=512,
        temperature=0.3,
        repetition_penalty=1.1,
        pad_token_id=128001
    )

    full_text = result[0]["generated_text"]
    return full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text

def generate_report(rank, timestamp):
    csv_path = get_latest_csv()
    cluster_info = get_cluster_info()
    module_versions = read_module_info()
    df, dept_usage_summary = analyze_data(csv_path)

    if rank == 0:
        audience = "internal"
        summary = generate_summary(df, cluster_info, dept_usage_summary, audience=audience)
        pdf_path = f"/data/researchHome/pdou/hpc_cluster_report/reports/internal_report_{timestamp}.pdf"
        create_pdf_report(cluster_info, df, summary, pdf_path, module_versions)
    elif rank == 1:
        audience = "external"
        summary = generate_summary(df, cluster_info, dept_usage_summary, audience=audience)
        pdf_path = f"/data/researchHome/pdou/hpc_cluster_report/reports/external_report_{timestamp}.pdf"
        create_pdf_report(cluster_info, df, summary, pdf_path, module_versions)

def main():
    dist.init_process_group(backend="gloo")               # gloo for CPU, NCCL for GPU
    rank = dist.get_rank()
    print(f"[Rank {rank}] running...")

    # 只允许前两个进程执行任务，其他直接退出
    if rank > 1:
        print(f"[Rank {rank}] skipping extra process")
        dist.destroy_process_group()
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generate_report(rank, timestamp)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
