# Week 4 - 數據修飾與使用TabNet

## Overview
- 把球種整合成7個種類，
- 對 hit area 做了 one-hot encoding
- 使用 imblearn 套件處理好 imbalance 的問題
- 使用 TabNet 訓練模型

## Files in this Folder
1. `01_shot_process.py`: 把最後五拍的資料整理出來
2. `02_combine_training.py`: 使用資料訓練模型，得到feature importance

## How to Run
1. 把需要的資料：'rally_\[winning, in, out\].csv' 和 'shot_grouped.csv' 放在資料夾底下
2. 下載需要的套件：pandas, chardet, sklearn, matplotlib, imblearn, pytorch_tabnet
3. 按照上面的順序執行程式

## Additional Notes
在 `02_combine_training.py`中，更改第16, 17行的變數值，可以分析不同得分方式：
- winning: 單純得分與否 (最後一拍得分為1)
- out: 出界得分 (最後一拍出界為1)
- in: 落地得分 (最後一拍得分為1)
註解與反註解第67-73, 74-80行，可以切換訓練的模型：
- 67-73: Random Forest
- 74-80: TabNet