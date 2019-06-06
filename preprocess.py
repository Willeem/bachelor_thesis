import re
from nltk.tokenize import RegexpTokenizer
from os import listdir, makedirs, getcwd
from os.path import isdir, isfile, join

def get_items_in_directory(filetype,directory):
    """Lists all items in a directory depending on filetype (directory or file)"""
    if filetype == 'directory':
        return [f for f in listdir(directory) if isdir(join(directory,f))]
    return [f for f in listdir(directory) if isfile(join(directory,f))]

def removeURL(comment):
    """Removes URLs from comments"""
    return re.sub(r'http\S+', '', comment)

def preprocess():
    """Removes punctuation and concatenates all comments into one big list of words"""
    teams = get_items_in_directory('directory','data/')
    for team in teams:
        if team not in ['Argentina','Australia','Belgium','Brazil','Colombia',
        'Croatia','England','France','Germany','Mexico','Poland','Portugal',
        'Republic_of_Ireland','Sweden','The_Netherlands','United_States']:
            files = get_items_in_directory('file','data/' + team)
            for user in files:
                with open('data/' + team + '/' + user, 'r', encoding='utf-8') as f:
                    data = f.read()
                    split_data = data.split('##########')
                    usable_comments = []
                    for comment in split_data:
                        comment = removeURL(comment)
                        tok_comment = RegexpTokenizer(r'\w+').tokenize(comment.strip().lower())
                        if len(tok_comment) > 0:
                            usable_comments.append(tok_comment)
                            usable_comments.append('##########')
                output_documents = [inner for outer in usable_comments for inner in outer]
                with open('data/' + team + '/' + user, 'w', encoding='utf-8') as outfile:
                    outfile.write(" ".join(output_documents))

preprocess()
