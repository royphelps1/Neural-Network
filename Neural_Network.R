# ~~ Roy Phelps ~~
# ~~ Using a Neural Network to determine if a cell is malignant or benign ~~
# ~~ July 15, 2023 ~~

# Initialize the packages
install.packages("neuralnet")
library("neuralnet")
install.packages("NeuralNetTools") # An alternative network visualization, predictor importance
library ("NeuralNetTools")


# Read the diabetes data file.  Change the file location for your file.
setwd("/Users/royphelps/Library/CloudStorage/OneDrive-UMGC/Classes/Data 630 9040/Weeks 8 and 9/Assignment 4")
di<-read.csv(file="wdbc.csv", head=TRUE, sep=",")


#~~~Exploratory Phase~~~
# Preview the first 6 rows and view the variables
head(di)
str(di)
summary(di)


#~~~Transformation and pre-processing phase~~~
# Transform the dependent variable to binary 0 = Benign, 1 = Malignant
# Replace "B" with 0 and "M" with 1 in the diagnosis column
di$diagnosis <- ifelse(di$diagnosis == "B", 0, 1)

# Remove variable ID
di$ID<-NULL


# Verify transformations
head(di)
table(di$diagnosis)

# Run the summary command for further verification of the transformations
summary(di)

# Scale the variables except for the dependent binary variable. 
str(di[2:31])
di[2:31]<-scale(di[2:31])


#~~~Model and algorithm intuition~~~
# Make sure that the result is reproducible
set.seed(12345)

# Split the data into a training and test set
ind <- sample(2, nrow(di), replace = TRUE, prob = c(0.7, 0.3))
train.data <- di[ind == 1, ]
test.data <- di[ind == 2, ]

# Build the model with additional model tuning and activation function 
nn<-neuralnet(formula = diagnosis~radius+texture+perimeter+area+smoothness+compactness+concavity+
                concave+symmetry+fractal+radius2+texture2+perimeter2+area2+smoothness2+compactness2+
                concavity2+concave2+symmetry2+fractal2+radius3+texture3+perimeter3+area3+smoothness3+
                compactness3+concavity3+concave3+symmetry3+fractal3, 
              data = train.data, 
              hidden=c(3), # Increase the number of hidden layers and neurons
                           # to capture more complex patterns in the data.
                           # Adds 1 hidden layer with 3 neurons 
              err.fct="ce", 
              act.fct="logistic",  # Activation function of sigmoid
              linear.output = FALSE, 
              rep = 3) # Apply L1 regularization to prevent over-fitting and 
                       # improve generalization. 


# Names command displays the available neural network properties
names(nn)

# Network properties
nn$call                  # the command we ran to generate the model
nn$response[1:10]        # actual values of the dependent variable for first 10 records
nn$covariate [1:20]      # input variables that were used to build the model for first 20 records
nn$model.list            # list dependent and independent variables in the model
nn$net.result[[1]][1:10] # display the first 10 predicted probabilities
nn$weights               # network weights after the last method iteration
nn$startweights          # weights on the first method iteration
nn$result.matrix         # number of training steps, the error, and the weights 


#~~~Visualization Phase~~~
# Visualizations
plot(nn,nodelabels=T)    # plot the network
plotnet(nn)
plotnet(nn, circle_col="yellow") #may change node color
# Relative importance for each variable; only for network with 1 hidden layer and one output
garson(nn)  
#Relative importance for each variable; the network may have >=1 hidden layers >=1 output
olden(nn)
plot(di$area3, di$perimeter3, xlab="Area", ylab="Perimeter", main="Relationship Between Area and Perimeter")

#~~~Models Performance Phase
# Model evaluation; Round the predicted probabilities
mypredict<-compute(nn, nn$covariate)$net.result
mypredict<-apply(mypredict, c(1), round)
mypredict [1:10]

# Confusion matrix for the training set
table(mypredict, train.data$diagnosis, dnn =c("Predicted", "Actual"))
mean(mypredict==train.data$diagnosis)

# Confusion matrix for the test set
testPred <- compute(nn, test.data[, 2:31])$net.result
testPred<-apply(testPred, c(1), round)
table(testPred, test.data$diagnosis, dnn =c("Predicted", "Actual"))
mean(testPred==test.data$diagnosis)

# Confusion matrix implementation in caret package
require(caret)
require(e1071)
confusionMatrix(table(testPred, test.data$diagnosis), dnn=c("predicted", "actual"))

# Create a subset from the test data to
# evaluate the accuracy of the neural network
# on a smaller portion of the test data
subset.data <- test.data[1:100, ]  # Change the subset size as needed

# Extract the independent variables from the subset data
subset.input <- subset.data[, 2:31]

# Predict using the neural network model on the subset data
subsetPred <- compute(nn, subset.input)$net.result
subsetPred <- apply(subsetPred, 1, round)

# Calculate accuracy on the subset data
accuracy <- mean(subsetPred == subset.data$diagnosis)
print(paste("Accuracy on the subset data:", accuracy*100,"%"))


# Confusion matrix for the training set
confusion_train <- table(mypredict, train.data$diagnosis)

# Confusion matrix for the test set
confusion_test <- table(testPred, test.data$diagnosis)

# Calculate precision, recall, and F1 score for the training set
tp_train <- confusion_train[2, 2] # Number of true positives for the training set
fp_train <- confusion_train[1, 2] # Number of false positives for the training set
fn_train <- confusion_train[2, 1] # Number of false negatives for the training set

precision_train <- tp_train / (tp_train + fp_train)
recall_train <- tp_train / (tp_train + fn_train)
f1_score_train <- 2 * (precision_train * recall_train) / (precision_train + recall_train)

# Calculate precision, recall, and F1 score for the test set
tp_test <- confusion_test[2, 2] # Correctly predicted as positive, malignant
fp_test <- confusion_test[1, 2] # Incorrectly predicted as positive, malignant
fn_test <- confusion_test[2, 1] # Incorrectly predicted as negative, benign

precision_test <- tp_test / (tp_test + fp_test)
recall_test <- tp_test / (tp_test + fn_test)
f1_score_test <- 2 * (precision_test * recall_test) / (precision_test + recall_test)
print(paste("F1 Score (Training):", f1_score_train*100,"%"))
print(paste("F1 Score (Test):", f1_score_test*100,"%"))






