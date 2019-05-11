# Author: Syed Aqhib Ahmed
# Date: 5/10/2019
# File: assign2.R

set.seed(1234)
rm(list = ls())
setwd("D://Syed//Graduate//Spring19//Optimization//nf//assignment2")

#
#install.packages("textfeatures")
#install.packages("textstem")
#install.packages("textrank")
#install.packages("tm")
#install.packages("gradDescent")
library(textfeatures)
library(textstem)
library(textrank)
library(tm)
library(gradDescent)
#Loading in the training data from the csv file to build the model
train_data <- read.csv("disaster_tweets.csv", header =  TRUE)

#Showing what the original data looks like before feature generation
head(train_data)

#Since we have collected the positive and negatice label data separately and concatenated them, it is important to shuffle the data
train_data <- train_data[sample(nrow(train_data)),]

#Shuffled data looks more natural
head(train_data)

#Feature Generating
feat <- textfeatures(train_data$text)
head(feat)
train_data$nlowers <- feat$n_lowers
train_data$nexclaims <- feat$n_exclaims

#Cleaning the data
train_data$text <- tolower(train_data$text)
stopwords_regex = paste(stopwords('en'), collapse = '\\b|\\b')
stopwords_regex = paste0('\\b', stopwords_regex, '\\b')
train_data$text = stringr::str_replace_all(train_data$text, stopwords_regex, '')

train_data$text <- gsub("[^ a-zA-Z']", " ", train_data$text)
train_data$text <- stemDocument(train_data$text)

text_source <- VectorSource(train_data$text)
text_corpus <- VCorpus(text_source)

TDM_text <- TermDocumentMatrix(text_corpus)
tweets_tdm_rm_sparse <- removeSparseTerms(TDM_text, 0.99)
TDM_text_matrix <- as.matrix(tweets_tdm_rm_sparse)
TDM_text_matrix <- (t(TDM_text_matrix))

dim(TDM_text_matrix)
head(TDM_text_matrix)

train_data <- cbind(train_data,TDM_text_matrix)

len <- sapply(strsplit(train_data$text, " "), length)
mean(len)

#Reorder Colmns so that Label is at the end
train_data <- train_data[c(4:88,3)]

## 75% of the sample size
smp_size <- floor(0.75 * nrow(train_data))


train_ind <- sample(seq_len(nrow(train_data)), size = smp_size)

train <- train_data[train_ind, ]
test <- train_data[-train_ind, ]

lin <- lm(Label~., data = train)
lin
pred <-predict(lin,test)

#Linear model accuract
pred <- ifelse(pred<0.5,0,1)
head(pred)
lin_acc <- 1-sum(test$Label-pred)/length(pred)

#SGD Model Accuracy
sgdModel <- SGD(train, alpha = 0.1, maxIter = 100, seed = NULL)
lin$coefficients <- sgdModel
pred <-predict(lin,test)
pred <- ifelse(pred<0.5,0,1)
sgd_acc <- 1-sum(abs(test$Label-pred))/length(pred)
sgd_acc

nesterov <- AGD(train, alpha = 0.1, maxIter = 100, momentum = 0.9, seed = NULL)
lin$coefficients <- nesterov
pred <-predict(lin,test)
head(pred)
pred <- ifelse(pred<0.5,0,1)
nest_acc <- 1-sum(abs(test$Label-pred))/length(pred)
nest_acc

#write.csv(train_data, file = "final_data.csv")
