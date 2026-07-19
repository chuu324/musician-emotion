"""
下载 MTG-Jamendo 子集音频（前 3 个 tar 包，~1.4 GB，552 首）
=====================================================
用法:
  python download_mtg_subset.py

流程:
  1. 下载前 3 个 tar 包 (tar-00, tar-01, tar-02)
  2. 解压所有音频到 ~/ddd/mtg_data/audio/
  3. 展平目录结构 (去掉子目录)
  4. 更新 JSON 元数据的 audio_path 字段
  5. 重新生成 train/val/test 划分
"""

import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

# ============================================================
# 配置
# ============================================================
BASE_URL = "https://cdn.freesound.org/mtg-jamendo/autotagging_moodtheme/audio-low"
TAR_NAMES = [f"autotagging_moodtheme_audio-low-{i:02d}.tar" for i in range(3)]

OUTPUT_DIR = Path(os.path.expanduser("~/ddd/mtg_data"))
AUDIO_DIR = OUTPUT_DIR / "audio"
TAR_DIR = OUTPUT_DIR / "tars"
METADATA_DIR = Path(os.path.expanduser("~/ddd/code/phase2_emotion_control/data"))


def download_tar(tar_name: str) -> bool:
    """下载单个 tar 包。"""
    tar_path = TAR_DIR / tar_name
    if tar_path.exists():
        print(f"  ✓ 已存在: {tar_name}")
        return True

    url = f"{BASE_URL}/{tar_name}"
    print(f"  下载 {tar_name} ...")
    print(f"  来自: {url}")
    try:
        urllib.request.urlretrieve(url, tar_path)
        size_mb = tar_path.stat().st_size / 1024 / 1024
        print(f"  ✓ 完成: {size_mb:.0f} MB")
        return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False


def extract_and_flatten(tar_name: str):
    """解压 tar 并展平目录结构。"""
    tar_path = TAR_DIR / tar_name
    if not tar_path.exists():
        return

    extract_dir = AUDIO_DIR / f"_tmp_{tar_name}"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_dir)

    # 展平：将子目录中的文件移到 audio 根目录
    count = 0
    for item in extract_dir.rglob("*.low.mp3"):
        # 相对路径中的目录前缀转成文件名前缀
        rel = item.relative_to(extract_dir)
        # e.g. "22/949222.low.mp3" → "22_949222.low.mp3"
        flat_name = str(rel).replace("/", "_")
        dest = AUDIO_DIR / flat_name
        if not dest.exists():
            shutil.move(str(item), str(dest))
            count += 1

    # 清理临时目录
    shutil.rmtree(extract_dir)
    return count


def main():
    os.makedirs(TAR_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # 1. 下载 tar 包
    print("=" * 50)
    print("1. 下载 tar 包")
    print("=" * 50)
    for tar_name in TAR_NAMES:
        download_tar(tar_name)

    # 2. 解压并展平
    print("\n" + "=" * 50)
    print("2. 解压 & 展平目录结构")
    print("=" * 50)
    total = 0
    for tar_name in TAR_NAMES:
        n = extract_and_flatten(tar_name)
        if n:
            print(f"  {tar_name}: {n} 个文件")
            total += n
    print(f"  共 {total} 首音频")

    # 3. 构建 path_in_tar → 实际文件路径 的映射
    print("\n" + "=" * 50)
    print("3. 建立文件映射")
    print("=" * 50)
    # 映射: "22/949222.low.mp3" → "/path/to/22_949222.low.mp3"
    path_map = {}
    for f in AUDIO_DIR.glob("*.low.mp3"):
        # 文件名 "22_949222.low.mp3" → 还原为 "22/949222.low.mp3"
        parts = f.stem.split("_", 1)  # ["22", "949222.low"]
        if len(parts) == 2:
            orig = f"{parts[0]}/{parts[1]}.mp3"  # "22/949222.low.mp3"
            path_map[orig] = str(f)
    print(f"  映射了 {len(path_map)} 个文件")

    # 4. 更新 JSON 元数据
    print("\n" + "=" * 50)
    print("4. 更新元数据 audio_path")
    print("=" * 50)
    for json_name in ["metadata_mtg.json", "train.json", "val.json", "test.json"]:
        json_path = METADATA_DIR / json_name
        if not json_path.exists():
            continue

        with open(json_path) as f:
            records = json.load(f)

        updated = 0
        for r in records:
            ap = r.get("audio_path", "")
            # 如果 audio_path 是 "22/949222.mp3" 格式（无目录前缀）
            # 或已经是完整路径但不存在的，尝试映射
            if ap and not os.path.isfile(ap):
                # 尝试匹配 path 格式: "22/949222.mp3" → 转为 "22/949222.low.mp3"
                for orig_path, real_path in path_map.items():
                    if ap.endswith(orig_path) or ap.replace(".mp3", ".low.mp3") == orig_path:
                        r["audio_path"] = real_path
                        updated += 1
                        break

        with open(json_path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"  {json_name}: 更新了 {updated}/{len(records)} 条")

    # 5. 统计结果
    print("\n" + "=" * 50)
    print("5. 统计")
    print("=" * 50)
    audio_count = len(list(AUDIO_DIR.glob("*.low.mp3")))
    tar_size = sum(p.stat().st_size for p in TAR_DIR.glob("*.tar")) / 1024 / 1024
    print(f"  音频文件: {audio_count} 个")
    print(f"  tar 包占用: {tar_size:.0f} MB")
    print(f"  音频目录: {AUDIO_DIR}")

    # 清理
    choice = input("\n是否删除 tar 包释放空间？(y/n): ").strip().lower()
    if choice == "y":
        shutil.rmtree(TAR_DIR)
        print("  ✓ tar 包已删除")

    print(f"\n✅ 完成！")
    print(f"下一步: 用真实数据训练")
    print(f"  /home/eir/ddd/.venv/bin/python train_adapter.py \\")
    print(f"      --data_mode demo --data_path ./data/train.json \\")
    print(f"      --epochs 30 --batch_size 4 --use_fidelity \\")
    print(f"      --save_path checkpoints/adapter_mtg_real.pth")


if __name__ == "__main__":
    main()
