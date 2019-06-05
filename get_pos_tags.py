from nltk import pos_tag
import pickle
from collections import defaultdict
from os import listdir, makedirs, getcwd
from os.path import isdir, isfile, join

def get_items_in_directory(filetype,directory):
    """Lists all items in a directory depending on filetype (directory or file)"""
    if filetype == 'directory':
        return [f for f in listdir(directory) if isdir(join(directory,f))]
    return [f for f in listdir(directory) if isfile(join(directory,f))]

def get_pos_tags(types):
    filename = 'pos_tags_' + types[0] + '.pickle'
    for type in types:
        labels = get_items_in_directory('directory',type)
        for team in labels:
            features = []
            files = get_items_in_directory('file',type + '/' + team)
            count = 0
            for file in files:
                feats = []
                with open(type + '/' + team + '/' + file) as f:
                    data = f.read()
                    for word in data.split():
                        feats.append(word)
                    features.append(pos_tag(feats))
            with open(filename,'wb') as f:
                pickle.dump(features,f)

def main():
    pos_tags_train = get_pos_tags(['train'])
    pos_tags_dev = get_pos_tags(['dev'])
    pos_tags_test = get_pos_tags(['test'])


if __name__ == "__main__":
    main()
