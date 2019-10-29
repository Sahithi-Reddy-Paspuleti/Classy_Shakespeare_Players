# Exploratory Data Analysis
### Number of lines each player has in a play
### Number of players in each play
### No of players in each play - Plot
### # of dialogues per each player
## # of dialogues per each player-Plot
# Feature Engineering
### Splitting the ActSceneLine in to three different columns
### Number of acts for each player
# Word vectorization
### Removing the punctuation marks in the player lines
### Lemmatization of words
### Word Tokenization
### Lets plot the wordcloud for sample of 20 players. Plotting them for the full player list take longer execution time
### TF-IDF vectorization
### Creating A data frame to fit into the model
### Classification Using Various models
### Accuracy Score with Naive Bayes, Decision Tree, Random Forest 
### Classification using Nearest Neighbour Classifier
### Let's see how the accuracy varies depending upon the max_depth of a decision tree
# Notes

- Performed EDA on the dataset
- Splitted the ActSceneLine Feature into 3 seperate columns to fit the model
- Coverted the PlayerLine into different Vectors using TFIDF vectorization
- Used Play, Act ,Scene, SceneLine, Player line columns to classify the players.
# observations

- Accuracy is more with the Decision Tree and Random Forest Classification Models
- The accuracy in for this data set is varying with the maxdif and mindif of the TFIDF vectorization.
- Accuracy is also increasing with increase in the max_depth parameter of the DecisionTree Classifier and becomes constant after a certain depth
