#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# 数据处理：选出速度column，去除\n，null替换为nan，,用上一时刻数据填补空缺,dtype改为float
# train data：去除最后两天，留最后一天作测试集
def get_data():
    df = pd.read_csv('46607520312.csv',engine='python')
    df.drop(['46607520312'],axis=1,inplace=True)
    df.drop(['0000'],axis=1,inplace=True)
    df.replace(to_replace=r"\n", value='', inplace=True, regex=True)
    df.replace(to_replace=r"null", value=np.nan, inplace=True, regex=True)
    df.fillna(method='pad',inplace=True)
    df = df.astype(float)
    return df.values


# 二次指数平滑(测试数据集一共2878个数据，取前2000个训练）
def main(input,a):
    # a = float(input())
    # a = np.array(0,1,0.2)
    x = input[:-1]
    # s1和s2分别为一次和二次的平滑值
    s1 = []
    s2 = []
    s1.append(x[0])
    s2.append(x[0])
    smooth1_t_1 = x[0]
    smooth2_t_1 = x[0]
    y=[x[0]]
    for t in range(1,len(input)-1):
        xt=x[t]
        smooth1_t=a*xt+(1-a)*smooth1_t_1
        smooth2_t=a*smooth1_t+(1-a)*smooth2_t_1
        s1.append(smooth1_t)
        s2.append(smooth2_t_1)
        smooth1_t_1=smooth1_t
        smooth2_t_1=smooth2_t
        at=2*smooth1_t-smooth2_t
        bt=a/(1-a)*(smooth1_t-smooth2_t)
        yt=at+bt
        y.append(yt)
    return y


# MSE计算
def error(x,y):
    x=x[1:]
    y=y[:]
    return mean_squared_error(x,y),mean_absolute_error(x,y)


if __name__ == '__main__':
    x=get_data()[:1440]
    #a=0.5
    y = main(x, 0.5)
    mse, mae = error(x, y)
    print('When a=0.5, MSE={:0.3f}'.format(mse) + ',MAE={:0.3f}'.format(mae))
    plt.plot(y, label='Predict' )
    plt.plot(x, label='Test Data')
    plt.xlabel('Data Number')
    plt.ylabel('Speed(km/h)')
    plt.title('Quadratic Smoothing Exponential model')
    plt.legend()
    plt.savefig('指数平滑法.png')
    plt.show()
