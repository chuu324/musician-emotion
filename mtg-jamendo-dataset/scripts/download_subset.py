"""下载挑选的子集：下载包含所需曲目的 tar 包，提取后清理"""
import csv
import os
import tarfile
import sys
import tempfile
import shutil
from pathlib import Path
from urllib.request import urlretrieve

# 配置
SUBSET_FILE = "data/my_subset_500.tsv"
OUTPUT_DIR = os.path.expanduser("~/ddd/mtg_data")
RAW_AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")  # 最终音频存放处
TAR_DIR = os.path.join(OUTPUT_DIR, "tars")          # tar 暂存
DATASET = "autotagging_moodtheme"
TYPE = "audio-low"
BASE_URL = f"https://cdn.freesound.org/mtg-jamendo/{DATASET}/{TYPE}"

os.makedirs(RAW_AUDIO_DIR, exist_ok=True)
os.makedirs(TAR_DIR, exist_ok=True)

# 1. 读取子集数据
with open(SUBSET_FILE) as f:
    reader = csv.DictReader(f, delimiter="\t")
    tracks = list(reader)

# 2. 找出每个 track 需要的 tar 包
#    path 格式: "22/949222.mp3" → tar 里是 "22/949222.low.mp3"
#    sha256_tracks.txt 中 "22/949222.low.mp3" 这个文件属于哪个 tar
sha256_tracks_file = f"data/download/{DATASET}_{TYPE}_sha256_tracks.txt"
sha256_tars_file = f"data/download/{DATASET}_{TYPE}_sha256_tars.txt"

# 读取所有 track 到 tar 的映射
track_to_tar = {}  # "22/949222.low.mp3" -> "autotagging_moodtheme_audio-low-00.tar"
with open(sha256_tracks_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            track_path = parts[1]
            track_to_tar[track_path] = None  # 先占位

# 读取 tar 列表（按顺序）
tar_names = []
with open(sha256_tars_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            tar_names.append(parts[1])

# 将 tracks.txt 按行对应到 tar
# sha256_tracks.txt 是按 tar 包顺序排列的，每个 tar 的内容连续
track_paths = list(track_to_tar.keys())
tar_index = 0
for i, track_path in enumerate(track_paths):
    if tar_index < len(tar_names):
        track_to_tar[track_path] = tar_names[tar_index]
    # 每首 track 换行，下一个 tar 包开始
    # 实际上 tracks.txt 是连续排列的，我们需要找到 tar 的分界点
    # 更准确的方式：从 tar 里读取文件列表
    # 简单方式：逐行分配
    # tracks.txt 里 track 按 tar 包分组排列，组间无分隔符，所以无法直接分界
    # 改用以下方式

# 更好的方法：直接按 tar 包下载并提取需要的文件
# 先确定哪些 tar 包包含我们需要的 track
needed_tars = set()
track_paths_in_tar = {}  # tar_name -> [track_paths]

# 把 path 从 "22/949222.mp3" 转为 "22/949222.low.mp3"
needed_track_paths_low = set()
for t in tracks:
    orig_path = t["path"]  # "22/949222.mp3"
    # 转为低质量文件名
    name = orig_path.replace(".mp3", ".low.mp3")
    needed_track_paths_low.add(name)

# 逐行扫描 sha256_tracks.txt，给 track 分配 tar 包
current_tar_idx = -1
with open(sha256_tracks_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            current_tar_idx += 1  # 空行可能是分隔符
            continue
        track_path = parts[1]
        if track_path in needed_track_paths_low:
            tar_name = tar_names[current_tar_idx] if current_tar_idx < len(tar_names) else tar_names[-1]
            needed_tars.add(tar_name)
            track_paths_in_tar.setdefault(tar_name, []).append(track_path)

# 因为 tracks.txt 是按 tar 顺序连续排列的，我们需要按 tar 边界来划分
# 简单重新来：用 tar 中文件列表来匹配
# 先下载 tar 包再提取
print(f"需要下载 {len(needed_tars)} 个 tar 包: {sorted(needed_tars)}")

# 3. 下载并提取
for tar_name in sorted(needed_tars):
    tar_url = f"{BASE_URL}/{tar_name}"
    tar_path = os.path.join(TAR_DIR, tar_name)

    if not os.path.exists(tar_path):
        print(f"\n下载 {tar_name} ...")
        try:
            urlretrieve(tar_url, tar_path)
            print(f"  ✓ {tar_name} 下载完成")
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            continue
    else:
        print(f"\n{tar_name} 已存在，直接解压")

    # 提取需要的文件
    print(f"  解压中 ...")
    extracted = 0
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            # 只提取我们需要的文件
            if member.name in needed_track_paths_low:
                member.name = os.path.basename(member.name)  # 去掉目录前缀
                tar.extract(member, path=RAW_AUDIO_DIR)
                extracted += 1
        print(f"  提取了 {extracted} 个文件到 {RAW_AUDIO_DIR}")

    # 删除 tar 包（省空间）
    os.remove(tar_path)
    print(f"  已删除 {tar_name}")

print(f"\n✅ 完成！音频保存在: {RAW_AUDIO_DIR}")
