sink('coin_classification.txt')

library(keras)
library(tidyverse)
library(tensorflow)
library(magick)
library(plyr)

coinFiles <- list.files("classification")

loadImg <- function(path) {
  coin <- image_read(paste0('classification/', path))
  coinResized <- image_resize(coin, "160x120")
  
  # Extract the raw bitmap matrix with pixel values with `image_data`.
  pix <- image_data(coinResized)
  
  # Convert it to an array.
  pix_arr <- as.integer(pix)
  
  # Flatten the array into a single dimension.
  array_reshape(pix_arr, 57600)
}

coinList <- lapply(coinFiles, loadImg)
length(coinList)

coins <- array(unlist(coinList), dim=c(3059, 57600))
dim(coins)

coinValues <- unlist(lapply(coinFiles, function (f) as.integer(unlist(strsplit(f,"_"))[1]))) 
unique(coinValues)

nTrain <- floor(0.90 * nrow(coins))
trainIndexes <- sample(1:nrow(coins), size = nTrain)
X_train <- coins[trainIndexes,]
Y_train <- coinValues[trainIndexes]
X_test <- coins[-trainIndexes,]
Y_test <- coinValues[-trainIndexes]

X_train_cen <- scale(X_train, scale=FALSE)
X_train_std <- scale(X_train)

tr_means <- colMeans(X_train)
X_test_cen <- sweep(X_test,2,tr_means,FUN="-")

reg <- regularizer_l2(l = 0.01)

model <- keras_model_sequential() %>%
  layer_dense(units=64, activation='relu', kernel_initializer='random_normal', kernel_regularizer = reg, input_shape = 57600) %>%
  layer_dense(units=128, activation='relu', kernel_initializer='random_normal', kernel_regularizer = reg,) %>%
  layer_dense(units=32, activation='relu', kernel_initializer='random_normal', kernel_regularizer = reg,)%>%
  layer_dense(units=64, activation='relu', kernel_initializer='random_normal', kernel_regularizer = reg,)%>%
  layer_dense(units=5 , activation='softmax', kernel_initializer='random_normal', kernel_regularizer=reg)

compile(model, loss = 'categorical_crossentropy', optimizer = optimizer_sgd(lr = 0.001), metrics = 'accuracy')
# compile(model, loss = "categorical_crossentropy", optimizer = optimizer_rmsprop(), metrics = "accuracy")

y = c(5, 10, 25, 50, 100)
y_map2 = c(0, 1, 2, 3, 4 )
y_map = cbind(y,y_map2)

y_map <- as.data.frame(y_map)

Y_train <- as.data.frame(Y_train)
colnames(Y_train) <- "y"
Y_test <- as.data.frame(Y_test)
colnames(Y_test) <- "y"

y_map_train <- join(Y_train, y_map)
y_map_test <- join(Y_test, y_map)

y_map_train <- y_map_train$y_map2

y_map_test <- y_map_test$y_map2


y_train_cat <- to_categorical(y_map_train)

y_test_cat <- to_categorical(y_map_test)


history <- fit(model,X_train_cen, y_train_cat, validation_split=0.20, epochs=50, batch_size=64)

plot(history)


evaluate(model,X_test_cen, y_test_cat)
evaluate
ModPred <- predict(model, X_test_cen)
ModPred

sink()
