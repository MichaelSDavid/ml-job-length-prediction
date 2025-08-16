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
model_info = {
    "resnet": {"model_name": "ResNet50", "data_name": "ImageNet"},
    "gpt3": {"model_name": "GPT3", "data_name": "C4"},
    "llama2_70b_lora": {"model_name": "Llama 2 70B", "data_name": "SCROLLS GovReport"},
    "stable_diffusion": {"model_name": "Stable Diffusion v2", "data_name": "LAION-400M-filtered"},
    "dlrm_dcnv2": {"model_name": "DLRM-dcnv2", "data_name": "Criteo 4TB multi-hot"},
    "ssd": {"model_name": "SSD", "data_name": "COCO"},
    "bert": {"model_name": "BERT-large", "data_name": "Wikipedia 2020/01/01"},
    "rgat": {"model_name": "R-GAT", "data_name": "IGBH-Full"},
    "unet3d": {"model_name": "3D U-Net", "data_name": "KiTS19"},
    "llama31_405b": {"model_name": "Llama3.1 405b", "data_name": "C4"},
    "retinanet": {"model_name": "RetinaNet", "data_name": "Open Images"}
}

# Utility to parse memory string
def parse_mem(mem_str):
    num, unit = re.findall(r"([\d.]+)\s*(GB|TB|MiB)", mem_str)[0]
    gb = float(num)
    if unit == "TB":
        gb *= 1000
    elif unit == "MiB":
        gb *= 0.001048576
    return gb

# Parse all system JSON files
records = []
for system_file in systems_dir.glob("*.json"):
    system_name = system_file.stem
    print(f"Looking at system: {system_name}")
    with open(system_file) as f:
        sys_data = json.load(f)
    
    try:
        nodes = int(sys_data["number_of_nodes"])
        # print(sys_data["number_of_nodes"])
        gpu_name = sys_data["accelerator_model_name"]
        gpu_count = int(sys_data["accelerators_per_node"]) * nodes
        gpu_mem_gb = parse_mem(sys_data["accelerator_memory_capacity"]) * nodes
        cpu_count = int(sys_data["host_processors_per_node"]) * nodes
        cpu_core_count = int(sys_data["host_processor_core_count"]) * nodes
        cpu_mem_gb = parse_mem(sys_data["host_memory_capacity"]) * nodes
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
        
        epoch_counts = []
        time_mins = []
        batch_size = None

        for result_file in model_folder.glob("*.txt"):
            print(f"Looking at result: {result_file}")
            with open(result_file) as f:
                lines = [line.strip() for line in f if line.startswith(":::MLLOG")]

            time_ms_list = []
            epoch_num = None
            epoch_occur = []

            for line in lines:
                try:
                    # print(f"line: {line.split(':::MLLOG')[1]}")
                    json_obj = json.loads(line.split(":::MLLOG")[1])
                except:
                    print("skipper 2?")
                    continue

                if "time_ms" in json_obj:
                    time_ms_list.append(json_obj["time_ms"])
                if "epoch_num" in line.split(":::MLLOG")[1]:
                    print("epoch_num present")
                    pattern = re.compile(r'"epoch_num"\s*:\s*(\d+)')
                    matches = pattern.findall(line)
                    if matches: epoch_occur.extend(int(m) for m in matches)
                    # epoch_occur.append(json_obj.get("metadata",{}).get("epoch_num"))
                if "global_batch_size" in line and batch_size is None:
                    batch_size = json_obj.get("value")

            if time_ms_list:
                total_time_min = (max(time_ms_list) - min(time_ms_list)) / (1000 * 60)
                time_mins.append(total_time_min)

            if epoch_occur:
                epoch_num = max(epoch_occur)
                epoch_counts.append(epoch_num)
            else:
                epoch_counts.append(0)

        if not time_mins or not epoch_counts or batch_size is None:
            print(f"time_mins: {time_mins}, epoch_counts: {epoch_counts}, batch_size: {batch_size}")
            continue

        avg_time_min = sum(time_mins) / len(time_mins)
        avg_epoch_count = math.ceil(sum(epoch_counts) / len(epoch_counts))

        record = {
            "model_name": model_name,
            "data_name": data_name,
            "gpu_name": gpu_name,
            "gpu_count": gpu_count,
            "cpu_count": cpu_count,
            "cpu_core_count": cpu_core_count,
            "cpu_mem_gb": cpu_mem_gb,
            "epoch_count": avg_epoch_count,
            "batch_size": batch_size,
            "time_min": avg_time_min,
            "gpu_mem_gb": gpu_mem_gb
        }
        records.append(record)

# Save to CSV
output_file = "mlperf_data.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    for r in records:
        writer.writerow(r)
