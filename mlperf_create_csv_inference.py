# MANUAL CORRECTIONS AND MERGING INVOLVED
import os
import json
import csv
import math
import re

from pathlib import Path

# Define the base directory
base_dir = Path("NVIDIA")
systems_dir = base_dir / "systems"
results_dir = base_dir / "results"

# Define model info mapping
# SQuad v1.1 and COCO-2014
model_info = {
    "3d-unet-99.9": {"model_name": "3D U-Net", "data_name": "KiTS19"},
    "3d-unet-99": {"model_name": "3D U-Net", "data_name": "KiTS19"},
    "bert-99.9": {"model_name": "BERT-large", "data_name": "SQuAD"},
    "bert-99": {"model_name": "BERT-large", "data_name": "SQuAD"},
    "dlrm-v2-99.9": {"model_name": "DLRM-dcnv2", "data_name": "Criteo 4TB multi-hot"},
    "dlrm-v2-99": {"model_name": "DLRM-dcnv2", "data_name": "Criteo 4TB multi-hot"},
    "gptj-99.9": {"model_name": "GPT-J 6B", "data_name": "CNN-DailyMail News Text Summarization"},
    "gptj-99": {"model_name": "GPT-J 6B", "data_name": "CNN-DailyMail News Text Summarization"},
    "resnet50": {"model_name": "ResNet50", "data_name": "ImageNet"},
    "rnnt": {"model_name": "RNNT", "data_name": "Librispeech"},
    "stable-diffusion-xl": {"model_name": "Stable Diffusion XL", "data_name": "COCO"},
    "retinanet": {"model_name": "RetinaNet", "data_name": "OpenImages"},
    "llama2-70b-99": {"model_name": "Llama 2 70B", "data_name": "OpenOrca"},
    "llama2-70b-99.9": {"model_name": "Llama 2 70B", "data_name": "OpenOrca"}
}

# Utility to parse memory string
def parse_mem(mem_str):
    num, unit = re.findall(r"([\d.]+)\s*(GB|TB)", mem_str)[0]
    gb = float(num)
    if unit == "TB":
        gb *= 1000
    return gb

# Parse all system JSON files
records = []
for system_file in systems_dir.glob("*.json"):
    system_name = system_file.stem
    print(f"Looking at system: {system_name}")
    with open(system_file) as f:
        sys_data = json.load(f)
    
    try:
        # ALL HAVE SAME NUMBER OF NODES (1)
        # print(sys_data["number_of_nodes"])
        gpu_name = sys_data["accelerator_model_name"]
        gpu_count = int(sys_data["accelerators_per_node"])

        if sys_data["accelerator_memory_capacity"] == "Shared with host":
            gpu_mem_gb = parse_mem(sys_data["host_memory_capacity"])
        else:
            gpu_mem_gb = parse_mem(sys_data["accelerator_memory_capacity"])
            
        cpu_count = int(sys_data["host_processors_per_node"])
        cpu_core_count = int(sys_data["host_processor_core_count"])
        cpu_mem_gb = parse_mem(sys_data["host_memory_capacity"])
    except KeyError:
        print("skipper 1?")
        continue  # Skip system if critical keys are missing

    result_model_dir = results_dir / system_name
    if not result_model_dir.exists():
        continue

    for model_folder in result_model_dir.iterdir():
        model_key = model_folder.name
        print(f"Looking at model: {model_key}")
        if model_key not in model_info:
            print("model_name skippers? should be only ds_store")
            continue

        model_name = model_info[model_key]["model_name"]
        data_name = model_info[model_key]["data_name"]
        
        # epoch_counts = []
        time_min = -1
        # scenario = "Offline"
        # batch_size = None

        model_dir = result_model_dir / model_key
        for scenario_folder in model_folder.iterdir():
            print(f"Scenario folder NAME: {scenario_folder.name}")
            if scenario_folder.name == "Offline":
                result_file = (((model_dir / "Offline") / "performance") / "run_1") / "mlperf_log_summary.txt"
                with open(result_file, 'r') as f:
                    lines = f.readlines()
                    mean_latency_ns = float(re.search(r':\s*(\d+)', [line for line in lines if "Mean latency (ns)" in line][0]).group(1))
                    samples_per_query = int(re.search(r':\s*(\d+)', [line for line in lines if "samples_per_query" in line][0]).group(1))
                    total_queries = math.ceil(24576 / samples_per_query)
                    time_min = (mean_latency_ns * total_queries) / (60e9)
                    # for line in f:
                    #     if "Mean latency (ns)" in line:
                    #         mean_latency = re.search(r':\s*(\d+)', line)
                    #         samples_per_query = re.search(r':\s*(\d+)', line)
                    #         num_queries = 
                    #         # time_min = float(match.group(1)) / (60 * 10**9)
                    #     # elif: 
            elif scenario_folder.name == "Server":
                continue
                # scenario = "Server"
                result_file = (((model_dir / "Server") / "performance") / "run_1") / "mlperf_log_summary.txt"
                with open(result_file, 'r') as f:
                    lines = f.readlines()
                    mean_latency_ns = float(re.search(r':\s*(\d+)', [line for line in lines if "Mean latency (ns)" in line][0]).group(1))
                    # samples_per_query = int(re.search(r':\s*(\d+)', [line for line in lines if "samples_per_query" in line]).group(1))
                    # total_queries = math.ceil(24576 / samples_per_query)
                    time_min = (mean_latency_ns * 270336) / (60e9)
                    # for line in f:
                    #     if "Mean latency (ns)" in line:
                    #         match = re.search(r':\s*(\d+)', line)
                    #         time_min = float(match.group(1)) / (60 * 10**9)
            elif scenario_folder.name == "SingleStream" or scenario_folder.name == ".DS_Store":
                continue

            # print(f"Scenario name: {scenario}")
            record = {
                "model_name": model_name,
                "data_name": data_name,
                "gpu_name": gpu_name,
                "gpu_count": gpu_count,
                "cpu_count": cpu_count,
                "cpu_core_count": cpu_core_count,
                "cpu_mem_gb": cpu_mem_gb,
                "epoch_count": 0,
                "batch_size": 0,
                "time_min": time_min,
                "gpu_mem_gb": gpu_mem_gb,
                "scenario": scenario_folder.name
            }
            records.append(record)


# Save to CSV
output_file = "mlperf_inf_data7.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    for r in records:
        writer.writerow(r)

# output_file
