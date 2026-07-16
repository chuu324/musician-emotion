# 大文件（GitHub Releases）

下载地址：
https://github.com/uknowsomeone2000/musician-emotioncontrol/releases/tag/v1.0-large-files

## 恢复 checkpoints（2 个分卷）

下载 checkpoints.part_aa、checkpoints.part_ab 后：

    cd archives
    cat checkpoints.part_* > checkpoints.zip
    unzip checkpoints.zip -d ..

## 恢复 data（3 个分卷）

下载 data_raw.part_aa、part_ab、part_ac 后：

    cd archives
    cat data_raw.part_* > data_raw.zip
    unzip data_raw.zip -d ..

## Git 仓已包含

代码、docs、62M data、v1/v2 checkpoint、eval 结果

## Git 仓未包含（Release 下载）

主 checkpoints/、完整 data_raw (3.8G)
