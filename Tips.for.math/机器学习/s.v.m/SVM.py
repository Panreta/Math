import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

# c = 10
result = []
def hyperplane(n):
    x1 = -5 + 10 * np.random.rand(n)
    y1 = 2 + 3 * x1 + 1 * x1 ** 2 #positive hyperplane
    return x1,y1

def Line_Plot(x,y):
    # 创建折线图
    plt.plot(x, y, label='Accuracy')  # 设置 x 和 y 数据，并为图例指定标签
    plt.xlabel('X-axis')  # 设置 X 轴标签
    plt.ylabel('Y-axis')  # 设置 Y 轴标签
    plt.title('Line Plot')  # 设置标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线

    # 显示图形
    plt.show()


def sVm(t,gamma):
    for i in range(t):
        # 生成数据
        np.random.seed(0) #test 0-9 the accuracy doesn't change
        n = 200
        x1,y1 = hyperplane(n)
        y1 += 10 * np.random.rand(n)#add bias
        x2,y2 = hyperplane(n)
        y2 -= 10 * np.random.rand(n)
        

        X_train = np.concatenate([np.column_stack((x1, y1)), np.column_stack((x2, y2))])#shape (400, 2)
        y_train = np.array([0] * n + [1] * n)# [0...0,1...1]
      
        # # 可视化训练数据
        # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)#generate 400 points, c gives the diiiferent color
        # plt.title('Training Data')
        # plt.figure()
        

        # 创建SVM分类器并训练
        svc_model = svm.SVC(kernel='rbf', gamma=gamma)#how to adjust gamma to increase the accuracy
        svc_model.fit(X_train, y_train)

        # 生成测试数据
        x1t,y1t = hyperplane(n)
        y1t += 10 * np.random.rand(n)
        x2t,y2t = hyperplane(n)
        y2t -= 10 * np.random.rand(n)

        X_test = np.concatenate([np.column_stack((x1t, y1t)), np.column_stack((x2t, y2t))])
        y_test = np.array([0] * n + [1] * n)

        # 预测并计算准确度
        y_pred = svc_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        result.append(accuracy)
        print(f'Accuracy: {accuracy}')

        # # 可视化测试数据及预测结果
        # plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Paired)
        # plt.title(f'Prediction - Accuracy: {accuracy}')
        # plt.show()


l = np.linspace(0,0.2,51)
 
for i in l:
    sVm(1,i)
    print(f"i = {i},")
    print()

Line_Plot(l,result)
    


# Basically, we need to enlarge the RBF to maximize the dual problem,but we need to know whether the data distribute near 0 or away from 0,
# which lead to shrink gamma or enlarge gamma , test shows  c = 10, gamma between [0.143,0.148] get the accuracy 0.95？