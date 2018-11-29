# sentiment_analysis
General objective: Sentiment analysis is to determine the overall contextual polarity (positive, negative) of a message. It is widely applied to reviews or comments on the internet for a variety of applications, ranging from marketing to custom services.  As iAdvize has integrated social media messages such as tweets, facebook comments and feedback etc), knowing the sentiment hidden behind them is important since it reflects customer's altitude towards some topics or products. The goal for us is therefore to create a tool that is able to evaluate the sentiment of messages and identify those with negative emotion such that consultants are able to provide instant and proper services to customers.

This repository aims to fulfilling this task. It contains mainly four parts including data, preprocessing, train and test. 

1, data: 1,600,000 tweets from sentiment140 dataset. To make it compatible with french, googletrans module in python has been used to translate them into french. Due to the volume limit of googletrans API, the french translation of data is not exhaustive. translation_to_fr.py is set to finish this purpose. 

2, preprocessing: aims to eliminate the noisy (unformal expression) in tweets by using regular expression.

3, train: selecting the best classification model from candidate models including Random Forest, Support Vector Machine, Bernoulli Naive Bayesian. The optimal values for hyperparameters are calibrated by RandomizedSearchCV model in sklean. The key idea of it is to use 5-Fold cross validation to obtain optimal combination of hyperparameters values. 

4, test: yielding accuracy comparison between models and an existing pattern module in python.

5, model: the best model.
