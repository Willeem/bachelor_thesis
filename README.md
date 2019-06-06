# Predicting the favourite football team of a Reddit user
A repository for the bachelor thesis of Willem Datema s2958058



## Packages needed
To make sure you have every package needed, create a new virtual environment, activate it and run the following command:

```
pip install -r requirements.txt
```
## Getting the data
To run the same experiment that I did, the steps for collecting the data can be skipped.
### Aquiring user names
Run the first script to gather usernames from the top 1000 posts of last month and all time.
After running this script you will have a pickle file called final_dict.pickle, containing the clubs that are represented by at least 100 users or more as keys and the username of their fans as values. There will also be a pickle called authors.pickle which contains a list of all the authors. The last thing this script does is create folders for the teams in the data/ directory.
The command for running this script is:
```
python3 get_data.py
```
### Aquiring comments
After this, the comments can be retrieved. By running this script, the folders created  by the last script get filled with the raw comments of a user. Their comments are seperated by 10 hashtags to be able to distinguish the comments later. Getting the comments can be done by running the following command:
```
python3 get_comments_per_user.py
```

## Preprocessing the data
After getting the data, it needs to be preprocessed. This can be done by running:
```
python3 preprocess.py
```
**WARNING!** This replaces the data in the /data/ folder by tokenized words, without making any distinctions per comment! Therefore this loses style information and the comments cannot be distinguished anymore.

## Analysing and splitting the data
To filter the last unwanted data out of the data set run:
```
python3 analysis_and_split.py
```
After doing this, you are left with a train/, dev/ and test/ folder filled files named after a user with
their comments already tokenised and splitted.
**IMPORTANT** If you have some country names left in the dev/ train/ and test/ folders, please remove these manually to conduct the same experiment I did.

## Classyfing
Finally, you are done preparing the data and you can start the classification experiments. The default settings
are that of the last run on the test set. Some of the other runs are commented out.
```
python3 classifier.py
```
A note with the function ngramsplitter from classifier.py is that this is only used when the data Xtrain is converted to a string first. This is needed for sklearns vectorizers to split the sentences into character ngrams. When using unigrams, make sure to either skip the converting of the data to strings or use the ngram_splitter as tokenizer instead of identity.
