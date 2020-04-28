#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Nadam
from keras.layers import LSTM
import datetime as dt

# 数据处理：选出速度column，去除\n，null替换为nan，,用上一时刻数据填补空缺,dtype改为float
# train data：去除最后两天，留最后一天作测试集
df = pd.read_csv('46607520312.csv',engine='python')
df.drop(['46607520312'],axis=1,inplace=True)
df.drop(['0000'],axis=1,inplace=True)
df.replace(to_replace=r"\n", value='', inplace=True, regex=True)
df.replace(to_replace=r"null", value=np.nan, inplace=True, regex=True)
df.fillna(method='pad',inplace=True)
df = df.astype(float)
train = df.loc[:len(df)-2880]
test = df.loc[len(df)-1440:]
'''
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])
plt.show()
'''
scaler = MinMaxScaler(feature_range=(-1, 1))
train_sc = scaler.fit_transform(train)
test_sc = scaler.transform(test)

# split:获取训练和测试数据。
X_train = train_sc[:-1]
y_train = train_sc[1:]
X_test = test_sc[:-1]
y_test = test_sc[1:]
# np.random.seed(7)

x_train=X_train.reshape(len(df)-2880,1,1)
x_test=X_test.reshape(1439,1,1)
time1 = dt.datetime.now()
# 模型构建
lstm_model = Sequential()
lstm_model.add(LSTM(1,activation='tanh',return_sequences=True, kernel_initializer='lecun_uniform'))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(1,activation='tanh', kernel_initializer='lecun_uniform'))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='Nadam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history_lstm_model = lstm_model.fit(x_train, y_train, epochs=5, batch_size=10, verbose=1, shuffle=False, callbacks=[early_stop])

y_pred_test_lstm = lstm_model.predict(x_test)
y_train_pred_lstm = lstm_model.predict(x_train)
inv_y_pre=scaler.inverse_transform(y_pred_test_lstm)
inv_y=scaler.inverse_transform(y_test)

print('MSE on train set is:{:0.3f}'.format(mean_squared_error(inv_y,inv_y_pre)))
print('MSE on test set is:{:0.3f}'.format(mean_squared_error(inv_y,inv_y_pre)))
# 模型保存
mp = "formal46607520312_EPOCH=5_batch=10.h5"
lstm_model.save(mp)

# 训练用时
time2 = dt.datetime.now()
print('time use:'+str(time2-time1))

# 结果反归一化&结果可视化，图片打印及保存

plt.plot(inv_y_pre,label='Predict')
# plt.plot(y_relu,label='Predict(relu)')
plt.plot(inv_y,label='Test data')
plt.legend()
# plt.title('LSTM model')
plt.ylabel('Speed')
plt.xlabel('Data Number')
plt.title('LSTM model')
plt.savefig('46607520312LSTM_epoch=5_batch=10.png')
plt.show()
