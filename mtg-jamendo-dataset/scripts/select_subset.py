"""从 MTG-Jamendo mood/theme 数据集中挑选子集，生成待下载列表"""
import sys
sys.path.insert(0, ".")

import commons

# 用官方解析器读取
tracks, tags, extra = commons.read_file("data/autotagging_moodtheme.tsv")
print(f"总数据量: {len(tracks)} 首")

# 按 mood 标签丰富度排序
track_list = []
for tid, info in tracks.items():
    mood_tags = info.get("mood/theme", set())
    track_list.append({
        "track_id": tid,
        "artist_id": info["artist_id"],
        "album_id": info["album_id"],
        "path": info["path"],
        "duration": info["duration"],
        "tags": ",".join(sorted(mood_tags)),
        "tag_count": len(mood_tags),
    })

track_list.sort(key=lambda x: x["tag_count"], reverse=True)
subset = track_list[:500]

# 保存
import csv

with open("data/my_subset_500.tsv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["track_id", "artist_id", "album_id", "path", "duration", "tags", "tag_count"], delimiter="\t")
    writer.writeheader()
    writer.writerows(subset)

print(f"挑了 {len(subset)} 首，已保存到 data/my_subset_500.tsv")
print(f"标签范围: {min(t['tag_count'] for t in subset)} ~ {max(t['tag_count'] for t in subset)} 个标签/首")
print("\n前 5 首预览:")
for t in subset[:5]:
    print(f"  {t['track_id']} | {t['path']} | {t['tag_count']}个标签: {t['tags']}")
