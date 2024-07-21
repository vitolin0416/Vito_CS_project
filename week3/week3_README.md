# Week 3 - 初步Feature analysis

## Overview
- 使用pandas把資料整理成我們需要的樣子，
- 擷取最後五拍，
- 將shot與rally的資料合併，
- 使用random forest找出影響勝率(或界內得分、界外得分)的feature。

## Files in this Folder
1. `shots_process.py`: 把最後五拍的資料整理出來
2. `shot_gouping.py`: 把每個五拍都擺到同一個row
3. `rally_process.py`: 把需要的rally資料整理出來
4. `rally_encode.py`: 把rally的致勝原因標記成0或1，藉此分析影響不同得分方式的feature
5. `_conbine_training.py`: 使用random forest做出feature analysis 

## How to Run
1. 把需要的資料：'rally_0108.csv' 和 'shot_drop_unwanted.csv' 放在資料夾底下
2. 下載需要的套件：pandas, chardet, sklearn, matplotlib
3. 按照上面的順序執行程式

## Additional Notes
在 `_combine_training.py`中，更改第5,6行的變數值，可以分析不同得分方式：
- winning: 單純得分與否 (最後一拍得分為1)
- out: 出界得分 (最後一拍出界為1)
- in: 落地得分 (最後一拍得分為1)
