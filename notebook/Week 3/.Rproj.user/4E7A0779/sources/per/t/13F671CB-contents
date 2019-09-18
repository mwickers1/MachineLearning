library(keras)
library(tidyverse)

model1 = keras_model_sequential() %>%   
  layer_dense(units = 64, activation = "tanh", kernel_initializer='random_normal', input_shape = 57600) %>%
  layer_dense(units=64, activation='tanh', kernel_initializer='random_normal') %>%
  layer_dense(units=64, activation='tanh', kernel_initializer='random_normal')%>%
  layer_dense(units=1, kernel_initializer='random_normal')

model2 <- keras_model_sequential() %>%
  layer_dense(units=64, activation='relu', kernel_initializer='random_normal', input_shape = 57600) %>%
  layer_dense(units=64, activation='relu', kernel_initializer='random_normal') %>%
  layer_dense(units=64, activation='relu', kernel_initializer='random_normal')%>%
  layer_dense(units=1, kernel_initializer='random_normal')

summary(model1)  
summary(model2)  

compile(model1, loss = "mse", optimizer = optimizer_sgd(lr = 0.0001))
compile(model2, loss = "mse", optimizer = optimizer_sgd(lr = 0.000001))

fit(model1,X_train, Y_train, epochs=10, batch_size=32)

evaluate(model1, X_test, Y_test)

predict(model1, X_test)

sum(Y_train)/length(Y_train)

fit(model1,X_train_cen, Y_train, epochs=10, batch_size=32)

fit(model2,X_train, Y_train, epochs=10, batch_size=32)

