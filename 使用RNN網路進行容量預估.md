# 使用RNN網路進行容量預估

[TOC]

## 研究動機
鋰離子 (Li-Ion) 電池是可充電電池，由於其化學能力，可以最大限度地延長電池壽命，同時提高功率能量密度。 由於這些原因，鋰離子電池廣受歡迎，並廣泛用於移動計算設備和汽車系統。 電池健康監測的一個基本參數是健康狀態 (SoH)，其根據最大可釋放容量計算得出，代表電池在能量存儲和輸送方面的功能，本次專題將會關注於循環 ANN（RNN）的實現，使用提供由美國國家航空航天局 (NASA) 提供的鋰離子電池數據集進行預測

## 軟硬體架構
* 訓練軟硬體
        *     i7-12750H+3060laptop
        *     pytorch 2.0.2+cuda 11.8

* 推論硬體
        *     jetson nano developer kit
        *    pytorch 1.13.2+cuda 10.2 
    ![](https://hackmd.io/_uploads/rJXI55Ww2.png)


## 資料集建置與預處理
本次資料集使用NASA提供的鋰離子電池數據集

首先將各cycle的電壓電流曲線經過Akima函數進行曲線擬和
![](https://hackmd.io/_uploads/HyPqD9bw3.png)
![](https://hackmd.io/_uploads/ryzwu9ZDn.png)
在使用Savitzky–Golay filter進行濾波接著再從中提取CCCT(定電流段的充電時間)與CVCT(定電壓段的充電時間)以及其餘固定電壓段的充電時間
![](https://hackmd.io/_uploads/rkqSWcbwh.png)
最後使用了以下特徵當作訓練資料集
* CCCT
* CVCT
* 3.7V~4.19V的充電時間
* 3.8V~4.19V的充電時間
* 3.7V~4.1V的充電時間
* 3.7V~4.1V的充電時間

![](https://hackmd.io/_uploads/H17kFcZPh.png)
可以觀察到這些特徵皆與電池容量呈現一定相關性

## 模型建置與預測結果

本次推論將會採用GRU、LSTM兩種模型進行訓練與比較
1. 輸入皆為取前三個cycle的各項特徵，並進行正規化
1. 當前的容量作為輸出
1. 訓練時會將各電池的特徵與容量曲線進行拼接以方便訓練
1. 權重值皆會經過正交初始化，bias的起始值設為0.1，cell state以及hidden state設為0
1. 使用MSE Loss 和Adam  Optimizer
* LSTM
![](https://hackmd.io/_uploads/ryHoiqWDh.png)

* GRU
![](https://hackmd.io/_uploads/rJhLAcbwn.png)

* 訓練效果
    * loss
        LSTM
![](https://hackmd.io/_uploads/Byv11sWw3.png)
GRU
![](https://hackmd.io/_uploads/Bk49yobDh.png)

    * train reslut
    LSTM
    ![](https://hackmd.io/_uploads/SyuGgo-w2.png)
    GRU
    ![](https://hackmd.io/_uploads/B1Lmes-wn.png)

    * test reult
    ![](https://hackmd.io/_uploads/HyWWOiWw3.png)
![](https://hackmd.io/_uploads/BkWZdiZv2.png)
![](https://hackmd.io/_uploads/rJgW-di-wh.png)
![](https://hackmd.io/_uploads/BkZbuibP2.png)
![](https://hackmd.io/_uploads/ByZbdsbD2.png)
![](https://hackmd.io/_uploads/BkWWui-v2.png)


# 結論

本次模型預測出來的曲線雖有呈現大致趨勢，但準確度依然有待改善，且預測結果皆有垂直分量的誤差，推斷有可能是因為各電池的特徵特性不同，以及特徵多樣性不足，導致有此種結果，未來會再增加特徵種類並試著使用不同的模型訓練，以比較各模型的成效差異
 
# 參考資料
資料級來源: https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset/code
參考論文:
https://ieeexplore.ieee.org/document/9133084




