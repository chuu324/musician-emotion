#!/bin/bash
# ==============================================================
# 下载 MTG-Jamendo 子集音频（只下载包含我们数据的 tar 包）
# ==============================================================
# 用法: bash download_mtg_subset.sh
# ==============================================================
set -e

MTG_REPO="$HOME/ddd/mtg-jamendo-dataset"
OUTPUT_DIR="$HOME/ddd/mtg_data"
AUDIO_DIR="$OUTPUT_DIR/audio"
SUBSET_FILE="$MTG_REPO/data/my_subset_500.tsv"
TRACKS_FILE="$MTG_REPO/data/download/autotagging_moodtheme_audio-low_sha256_tracks.txt"
TARS_FILE="$MTG_REPO/data/download/autotagging_moodtheme_audio-low_sha256_tars.txt"

mkdir -p "$AUDIO_DIR"

echo "========================================"
echo "1. 分析子集分布在哪些 tar 包中"
echo "========================================"

# 读取子集的 track_id
python3 << 'PYEOF'
import csv, os
from collections import defaultdict

subset_path = os.path.expanduser("$SUBSET_FILE")
tracks_path = os.path.expanduser("$TRACKS_FILE")
tars_path = os.path.expanduser("$TARS_FILE")

# 用实际路径替换变量
subset_path = "/home/eir/ddd/mtg-jamendo-dataset/data/my_subset_500.tsv"
tracks_path = "/home/eir/ddd/mtg-jamendo-dataset/data/download/autotagging_moodtheme_audio-low_sha256_tracks.txt"
tars_path = "/home/eir/ddd/mtg-jamendo-dataset/data/download/autotagging_moodtheme_audio-low_sha256_tars.txt"

# 读取子集
track_ids = set()
with open(subset_path) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        track_ids.add(row["track_id"])

print(f"子集共 {len(track_ids)} 首")

# 读取 tar 列表
tar_names = []
with open(tars_path) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            tar_names.append(parts[1])
print(f"共 {len(tar_names)} 个 tar 包")

# 读取所有 track 并分配 tar
track_tar_map = {}  # track_path -> tar_name
current_tar_idx = -1
lines_processed = 0

with open(tracks_path) as f:
    # 按行数平均分配 tar
    all_lines = f.readlines()
    lines_per_tar = len(all_lines) // len(tar_names)
    
    for i, line in enumerate(all_lines):
        parts = line.strip().split()
        if len(parts) >= 2:
            tar_idx = min(i // lines_per_tar, len(tar_names) - 1)
            track_path = parts[1]  # e.g. "48/948.low.mp3"
            track_tar_map[track_path] = tar_names[tar_idx]

# 匹配我们的子集
needed_tars = defaultdict(list)
matched = 0
unmatched = []

# 从子集中读取路径
with open(subset_path) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        orig_path = row["path"]  # "22/949222.mp3"
        low_path = orig_path.replace(".mp3", ".low.mp3")
        
        if low_path in track_tar_map:
            tar = track_tar_map[low_path]
            needed_tars[tar].append(low_path)
            matched += 1
        else:
            unmatched.append(orig_path)

print(f"匹配成功: {matched} 首")
print(f"未匹配: {len(unmatched)} 首")

# 输出需要下载的 tar
print(f"\n需要下载的 tar 包: {len(needed_tars)} 个")
total_size_gb = 0
for tar in sorted(needed_tars.keys()):
    count = len(needed_tars[tar])
    size_gb = count / 4000 * 4.6  # 估算：每个 tar ~4.6GB，~1850 tracks
    total_size_gb += size_gb
    print(f"  {tar}: {count} 首 (约 {size_gb:.1f} GB)")

print(f"\n预估总下载量: {total_size_gb:.1f} GB")

# 保存需要提取的文件列表
extract_list_path = "/tmp/mtg_extract_list.txt"
with open(extract_list_path, "w") as f:
    for tar in sorted(needed_tars.keys()):
        for path in needed_tars[tar]:
            f.write(f"{tar}|{path}\n")
print(f"提取列表已保存到: {extract_list_path}")

# 保存需要下载的 tar 列表
tar_list_path = "/tmp/mtg_tar_list.txt"
with open(tar_list_path, "w") as f:
    for tar in sorted(needed_tars.keys()):
        f.write(f"{tar}\n")
print(f"下载列表已保存到: {tar_list_path}")

PYEOF

echo ""
echo "========================================"
echo "2. 下载需要的 tar 包"
echo "========================================"

# 读取需要下载的 tar 列表
TAR_LIST=$(cat /tmp/mtg_tar_list.txt)

source "$HOME/ddd/mtg-jamendo-dataset/venv/bin/activate"

for TAR in $TAR_LIST; do
    TAR_PATH="$OUTPUT_DIR/$TAR"
    if [ -f "$TAR_PATH" ]; then
        echo "已存在，跳过: $TAR"
    else
        echo "下载 $TAR ..."
        python3 -c "
import requests
from tqdm import tqdm
import os

url = f'https://cdn.freesound.org/mtg-jamendo/autotagging_moodtheme/audio-low/$TAR'
output = '$TAR_PATH'
print(f'下载: {url}')
print(f'保存到: {output}')

res = requests.get(url, stream=True)
total = int(res.headers.get('Content-Length', 0))

with open(output, 'wb') as f:
    with tqdm(total=total, unit='B', unit_scale=True) as pbar:
        for chunk in res.iter_content(chunk_size=512*1024):
            f.write(chunk)
            pbar.update(len(chunk))
print(f'下载完成: $TAR')
"
    fi
done

echo ""
echo "========================================"
echo "3. 解压并提取需要的音频文件"
echo "========================================"

# 读取提取列表
while IFS='|' read -r TAR TRACK_PATH; do
    # 目标文件名：去掉目录前缀，保持唯一
    BASENAME=$(basename "$TRACK_PATH")  # "949222.low.mp3"
    # 同时保留目录前缀避免重名
    DIR_PREFIX=$(dirname "$TRACK_PATH")  # "22"
    DEST_NAME="${DIR_PREFIX}_${BASENAME}"  # "22_949222.low.mp3"
    DEST_PATH="$AUDIO_DIR/$DEST_NAME"
    
    if [ -f "$DEST_PATH" ]; then
        continue
    fi
    
    TAR_PATH="$OUTPUT_DIR/$TAR"
    if [ -f "$TAR_PATH" ]; then
        echo "提取: $TRACK_PATH"
        tar -xf "$TAR_PATH" -C "$AUDIO_DIR" --strip-components=1 "$TRACK_PATH" 2>/dev/null || \
        tar -xf "$TAR_PATH" -O "$TRACK_PATH" > "$DEST_PATH" 2>/dev/null || \
        echo "  跳过 $TRACK_PATH (解压失败)"
    fi
done < /tmp/mtg_extract_list.txt

# 重命名文件：去掉.low后缀，统一为.mp3
echo "统一文件名 ..."
for f in "$AUDIO_DIR"/*.low.mp3; do
    [ -f "$f" ] && mv "$f" "${f%.low.mp3}.mp3"
done

echo ""
echo "========================================"
echo "4. 统计下载结果"
echo "========================================"
COUNT=$(ls "$AUDIO_DIR"/*.mp3 2>/dev/null | wc -l)
echo "共下载并提取 $COUNT 个音频文件"
echo "音频目录: $AUDIO_DIR"

# 清理 tar 包（省空间）
echo ""
read -p "是否删除下载的 tar 包以释放空间？(y/n): " CLEANUP
if [ "$CLEANUP" = "y" ]; then
    while IFS='|' read -r TAR _; do
        rm -f "$OUTPUT_DIR/$TAR"
        echo "已删除: $TAR"
    done < <(sort -u /tmp/mtg_extract_list.txt)
    echo "清理完成！"
fi

echo ""
echo "✅ 完成！音频保存在: $AUDIO_DIR"
echo "下一步: 在 train_adapter.py 中 --data_mode demo --data_path ./data/train.json 使用真实音频训练"
