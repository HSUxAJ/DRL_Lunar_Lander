pip install -r requirements.txt

python main.py --episodes 5000 --lr 0.01 --gamma 0.99 --batch 10

python main.py --checkpoint ./checkpoints/policy_model-1000.pth

python main.py --checkpoint ./checkpoints/policy_model-1000.pth --render

| Reward 區間 | 評估輸出         |
| --------- | ------------ |
| ≥ 200     | `successful` |
| 50 \~ 199 | `land`       |
| < 50      | `fail`       |

