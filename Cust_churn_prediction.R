# Churn Prediction Comparing Logistic Regression, Decision Tree and Random Forest
# For the Classification Problem
# Data from the Telcom Churn Dataset
# https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets?resource=download
library(plyr)
library(corrplot)
library(ggplot2)
library(gridExtra)
library(ggthemes)
library(caret)
library(MASS)
library(randomForest)
library(party)
library(dplyr)

##load data
churn1 <- read.csv('churn-bigml-80.csv')
churn2 <- read.csv('churn-bigml-20.csv')
churn <- rbind(churn1,churn2)
str(churn)
# turn required variables to factor
churn$Churn <- as.factor(churn$Churn)
churn$State <- as.factor(churn$State)
churn$International.plan <- as.factor(churn$International.plan)
churn$Voice.mail.plan <- as.factor(churn$Voice.mail.plan)

# check for missing values in each column
sapply(churn, function(x) sum(is.na(x)))
# result:  no missing values for any column

#remove rows with missing values in any column of data frame
churn <- churn[complete.cases(churn), ]

## EDA
# Correlation between numeric variables
numeric.var <- sapply(churn, is.numeric)
corr.matrix <- cor(churn[,numeric.var])
# corrplot(corr.matrix, main="\n\nCorrelation Plot for Numerical Variables", method="number")

# Lower and upper triangular part of a correlation matrix
lower.tri((churn[,numeric.var]), diag = FALSE)
upper.tri((churn[,numeric.var]), diag = FALSE)

# Hide upper trangle
upper<-corr.matrix
upper[upper.tri(corr.matrix)]<-""
upper<-as.data.frame(upper)
upper
# write all matrix to CSV
write.csv(upper, "correlation results_numeric_variables.csv")

## Extract important correlations
library(reshape2)
# coeff greater than 0.30
subsetg030 <- subset(melt(corr.matrix),value>.30)
write.csv(subsetg030, "correlation results_numeric_variablescoeffg030.csv")
# list of those with coeff greater than 0.30
# coeff less than -0.30
subsetl030 <- subset(melt(corr.matrix),value < -.30)
write.csv(subsetl030, "correlation results_numeric_variablescoeffl030.csv")
# empty CSV file

# from the results (check of the correlation matrix created above) 
# the total charges and the total minutes are higly correlated
# so we will remove the charges from the dataset

## remove redundant variables

churn <- churn %>% select(-contains(".charge")) 

## Explore the categorical variables

p1 <- ggplot(churn, aes(x=International.plan)) + ggtitle("Intl_Plan") + xlab("Intl_Plan") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentage") + coord_flip() + theme_minimal()
p2 <- ggplot(churn, aes(x=Voice.mail.plan)) + ggtitle("Voice Mail Plan") + xlab("Voice Mail Plan") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentage") + coord_flip() + theme_minimal()
p3 <- ggplot(churn, aes(x=Churn)) + ggtitle("Churn") + xlab("Churn") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), width = 0.5) + ylab("Percentage") + coord_flip() + theme_minimal()
grid.arrange(p1, p2, p3, ncol=2)


## split the data into training and testing sets
intrain<- createDataPartition(churn$Churn,p=0.8,list=FALSE)
set.seed(127)
training<- churn[intrain,]
testing<- churn[-intrain,]

## Prediction

## Fitting the Logistic Regression Model:
LogModel <- glm(Churn ~ .,family=binomial(link="logit"),data=training)
print(summary(LogModel))



#Checking the variance table for examination of feature importance
anova(LogModel, test="Chisq")
# results - order of importance
# Customer.service.calls > total.intl.call > total.intl.minutes >
# total.night.minutes > total.eve.minutes > total.day.minutes >
# no.vmail.messages > voice.mail.plan > intl.plan > State
# thus, State is the least important among the significant variables

#Assessing the predictive ability of the Logistic Regression model

testing$Churn <- as.character(testing$Churn)
testing$Churn[testing$Churn=="False"] <- "0"
testing$Churn[testing$Churn=="True"] <- "1"
fitted.results <- predict(LogModel,newdata=testing,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != testing$Churn)
print(paste('Logistic Regression Accuracy',1-misClasificError))
# [1] "Logistic Regression Accuracy 0.854354354354354"

# Logistic Regression Confusion Matrix

print("Confusion Matrix for Logistic Regression");
table(testing$Churn, fitted.results > 0.5)

# Odd ratios computation i.e. what is the odds of an churn happening
library(MASS)
exp(cbind(OR=coef(LogModel), confint(LogModel)))
# Intl plan, Intl call, voice mail plan and State VT greatly increase the odds
# of Churn 8 - 9 folds

# Lets try the Decision Tree Model
# using those variable which increased the odds the most for simplicity sake
tree <- ctree(Churn~State+International.plan+Voice.mail.plan+
                Total.intl.calls, training)
plot(tree)

# create similar factor for the testing data
# testing$State <- as.factor(testing$State)
# testing$International.plan <- as.factor(testing$International.plan)
# testing$Voice.mail.plan <- as.factor(testing$Voice.mail.plan)

# Decision Tree Confusion Matrix
pred_tree <- predict(tree, testing)
print("Confusion Matrix for Decision Tree"); 
table(Predicted = pred_tree, Actual = testing$Churn)

# Decision Tree Accuracy

p1 <- predict(tree, training)
tab1 <- table(Predicted = p1, Actual = training$Churn)
tab2 <- table(Predicted = pred_tree, Actual = testing$Churn)
print(paste('Decision Tree Accuracy',sum(diag(tab2))/sum(tab2)))
# [1] "Decision Tree Accuracy 0.0.866366366366366"

# Can Random Forest do better let's find out
# Random Forest First Model

rfModel <- randomForest(Churn ~., data = training)
print(rfModel)
# Result 30% when predicting Churn and 1% when predicting non-churn

# Random Forest Prediction and Confusion Matrix
testing$Churn <- as.character(testing$Churn)
testing$Churn[testing$Churn=="0"] <- "False"
testing$Churn[testing$Churn=="1"] <- "True"
testing$Churn <- as.factor(testing$Churn)
pred_rf <- predict(rfModel, testing)
caret::confusionMatrix(pred_rf, testing$Churn)

# Random Forest Error Rate

plot(rfModel)

## Lets tune the Random Forest Model
library(doParallel)
cores <- makeCluster(detectCores()-1)
registerDoParallel(cores = cores)

control <- trainControl(method = 'repeatedcv',
                        number = 10,
                        repeats = 3,
                        search = 'grid')
# Error rate stabilised at around 300 trees from the plot

# Lets Tune ntree while holding mtry constant

#create tunegrid
tunegrid <- expand.grid(.mtry = c(sqrt(ncol(training))))
modellist <- list()

#train with different ntree parameters

for (ntree in c(200,300,400, 500, 600, 1000,200)){
  set.seed(123)
  fit <- train(Churn~.,
               data = training,
               method = 'rf',
               metric = 'Accuracy',
               tuneGrid = tunegrid,
               trControl = control,
               ntree = ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}
stopCluster(cores)
#Compare results
results <- resamples(modellist)
summary(results)
dotplot(results)
## Result 
# Our model have highest accuracy at ntree = 200 when accuracy = 88.88945% (Mean).
# Now we that we know the number of trees lets find the optimum mtry
max(fit$results$Accuracy)
####
bestmtry <- tuneRF(training[, -16], training[, 16], stepFactor = 0.5, plot = TRUE, 
            ntreeTry = 200, trace = TRUE, improve = 0.01)

# This plot give us some ideas on the number of mtry to choose. 
# OOB error rate is at the lowest when mtry is 6. Therefore, we choose mtry = 6.

# Fit the Random Forest Model After Tuning
rfModel_new <- randomForest(Churn ~., data = training, ntree = 200, mtry = 6, 
                            importance = TRUE, proximity = TRUE)
print(rfModel_new)

# OOB error rate increased to 6.07% from 5.85%
# but Class error improved 28.1% for Churn compared to the initial 30.7% (training dataset)
# Class Error for non-churn increased from 1.36% to 2.32% 

# Random Forest Predictions and Confusion Matrix After Tuning
pred_rf_new <- predict(rfModel_new, testing)
caret::confusionMatrix(pred_rf_new, testing$Churn)

# the improved and the initial mode have similar accuracy for the testing dataset
# specificity improved for the improved model from 64% to 78%
# sensitivity declined from 98.9% (initial) and 97.37% (improved)

# Which variables are the most valuable for teh model
# Random Forest Feature Importance
varImpPlot(rfModel_new, sort=T, n.var = 10, main = 'Top 10 Feature Importance')

# MDA - total Day minutes, customer service calls and international play most important
# MDG -however, in addition to total day minutes and customer service calls
# State is also a good variable for separating Churn from Non-churn customers

# Summary
# Random forest performed better than Decision Tree and Logistic regression