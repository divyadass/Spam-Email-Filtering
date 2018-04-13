emails = read.csv('energy_bids.csv', stringsAsFactors = FALSE)
str(emails)
emails$email[1]
table(emails$responsive)

### Now that we have our data, it's time to construct and pre-process the corpus
library(tm)
corpus = Corpus(VectorSource(emails$email))
strwrap(corpus[[1]]$content)

# convert to lower case
corpus = tm_map(corpus, FUN = tolower)

# remove puntuation
corpus = tm_map(corpus, removePunctuation)

# remove stop words
corpus = tm_map(corpus, removeWords, stopwords('en'))

#  Stemming the document
corpus = tm_map(corpus, stemDocument)

strwrap(corpus[[1]]$content)

### with the pre-processed corpus, it's time to build document-term matrix
### for our corpus
dtm = DocumentTermMatrix(corpus)
dtm
### removing sparse terms
dtm = removeSparseTerms(dtm, 0.97)
dtm

## now lets build a dataframe out of this document term matrix
labeledTerms = as.data.frame(as.matrix(dtm))
labeledTerms$responsive = emails$responsive 
str(labeledTerms)

### Now it's time to split our data into training and testing set and actually
### build the model
library(caTools)
set.seed(144)
split = sample.split(labeledTerms$responsive, SplitRatio = 0.7)
train = subset(labeledTerms , split == TRUE)
test = subset(labeledTerms, split == FALSE)
## building CART model
library(rpart)
library(rpart.plot)
emailCART = rpart(responsive ~ ., data = train, method = 'class')
prp(emailCART)
# let us evaluate model on test set
pred = predict(emailCART, newdata = test)
pred[1:10,]
pred.prob = pred[,2]
table(test$responsive, pred.prob >= 0.5)
## Accuracy comes out to be 0.856

##lets have  a look at baseline model
table(test$responsive)
## Accuracy is 0.83

### let us look at the performance of our model with ROC curve
library(ROCR)
predROCR = prediction(pred.prob, test$responsive)
perfROCR = performance(predROCR, 'tpr', 'fpr')
plot(perfROCR, colorize = TRUE)
## from the plot,  we saw that a threshold of around 0.15 would be good if we 
## want to favour false postives.

#### let's also compute the AUC value with the help of ROCR package
performance(predROCR, 'auc')@y.values
# So, we have an AUC of 0.79 in the test set. This means that our model can 
# differentiate between randomlt selected responsice and non-responsive 
# document 80% of the time.
