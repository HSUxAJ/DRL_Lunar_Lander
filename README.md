# 🛰️ LunarLander Policy Gradient (REINFORCE)

使用 Policy Gradient (REINFORCE) 方法訓練 `LunarLander-v3` 環境的智能體。支援訓練、模型儲存、評估模式、訓練過程圖像輸出等功能。

---

##  安裝依賴

```bash
pip install -r requirements.txt
```

---

##  訓練指令
```bash
python main.py --episodes 5000 --lr 0.01 --gamma 0.99 --batch 10
```

---

## 執行指令
```bash
python main.py --checkpoint ./checkpoints/policy_model-1000.pth
```
加上渲染
```bash
python main.py --checkpoint ./checkpoints/policy_model-1000.pth --render
```

---

## 遊戲分數

| Reward 區間 | 評估輸出         |
| --------- | ------------ |
| ≥ 200     | `successful` |
| 50 \~ 199 | `land`       |
| < 50      | `fail`       |

