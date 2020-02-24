# Double-DQN & DUELING-DQN CODE INTRODUCTION

## 文件介绍：

+ ### 环境为Maze_env
+ ### 神经网络结构为RL_brain_Double & RL_brain_Dueling
+ ### Q-learning算法为Training_model

## 使用技巧：

+ ### 经验回放
+ ### 延迟更新
+ ### Double-DQN将动作估计和动作选择分开
+ ### Dueling-DQN将动作估计值Q拆分为状态值V+优势值A：
  + 其中优势值A为了有唯一值，即辨识度使用和平均或者最大值的偏差作为优势
  + 详细见survy