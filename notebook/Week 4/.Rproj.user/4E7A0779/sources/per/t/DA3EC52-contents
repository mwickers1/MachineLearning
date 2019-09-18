library(ggplot2)
library(dbscan)

iris <- read.csv('iris.csv', stringsAsFactors = F)

set.seed(20)
clusters <- kmeans(iris[,3:4], 3, nstart = 20)
clusters$cluster
clusters$centers

clusters$cluster <- as.factor(clusters$cluster)
ggplot(iris, aes(Petal.Length, Petal.Width, color = clusters$cluster)) + geom_point()

dbscan = dbscan(iris[,3:4],eps = 1, minPts = 5 )
dbscan$cluster

d <- dist(iris[,3:4], method = "euclidean")
hc1 <- hclust(d, method = "complete" )
plot(hc1, cex = 0.6, hang = -1)
