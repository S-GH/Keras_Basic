#1. 데이터
import numpy as np
x_train= np.array([1,2,3,4,5,6,7,8,9,10]) 
y_train= np.array([1,2,3,4,5,6,7,8,9,10])
x_test= np.array([11,12,13,14,15,16,17,18,19,20]) 
y_test= np.array([11,12,13,14,15,16,17,18,19,20]) 


# print(x.shape) 
# print(y.shape)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim =1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()


#3.훈련
model.compile(loss ='mse', optimizer ='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs =200, batch_size=1)

#4. 평가
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse: ', mse)

x_prd = np.array([11,12,13])
aaa= model.predict(x_prd, batch_size=1)
print(aaa)

# bbb = model.predict(x, batch_size=1)
# print(bbb)


