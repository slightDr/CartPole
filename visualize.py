import matplotlib.pyplot as plt


scores = [9, 21, 11, 20, 10, 17, 15, 12, 9, 14, 9, 20, 13, 17, 11, 9, 9, 10, 9, 9, 9, 9, 9, 11, 9, 11, 9, 10, 10, 9, 10, 12, 11, 10, 10, 14, 10, 11, 28, 10, 10, 10, 12, 10, 10, 12, 12, 13, 21, 19, 64, 18, 67, 26, 31, 26, 27, 116, 57, 35, 36, 58, 36, 31, 33, 55, 41, 128, 40, 48, 62, 40, 61, 105, 94, 38, 93, 190, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]


# 绘制折线图
plt.figure(figsize=(10, 5))  # 设置图形的大小
plt.plot(scores, marker='o')  # 绘制折线图，'o'表示点的形状
plt.title('Scores Line Chart')  # 图形的标题
plt.xlabel('Index')  # x轴的标签
plt.ylabel('Score')  # y轴的标签
plt.grid(True)  # 显示网格
plt.show()  # 显示图形