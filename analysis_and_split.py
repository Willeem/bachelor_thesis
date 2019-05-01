#!/usr/bin/python3

from os import listdir, makedirs, getcwd
from os.path import isdir, isfile, join
from collections import defaultdict
from shutil import copy2


def get_items_in_directory(filetype,directory):
    """Lists all items in a directory depending on filetype (directory or file)"""
    if filetype == 'directory':
        return [f for f in listdir(directory) if isdir(join(directory,f))]
    return [f for f in listdir(directory) if isfile(join(directory,f))]


def count_words(data):
    word_count = 0
    for comment in data:
        comment = comment.split()
        word_count += len(comment)
    return word_count


def analysis():
    teams = get_items_in_directory('directory','data/')
    users_after_filters = defaultdict(list)
    for team in teams:
        users = 0
        word_count = 0
        files = get_items_in_directory('file','data/' + team)
        for user in files:
            data = open('data/' + team + '/' + user, 'r', encoding='utf-8').read()
            split_data = data.split('##########')
            word_count = count_words(split_data)
            if len(files) > 2000:
                #if word_count > 2000:
                users += 1
                users_after_filters[team].append(user)
                if users == 2000:
                    break
        print(team,users)
    return users_after_filters


def write_to_directory(item,items):
    for type in items:
        outpath = getcwd() + '/' + type + '/' + item
        makedirs(outpath)
        for file in items[type]:
            copy2('data/' + item + '/' + file, type + '/' + item)

def split_train_test_dev(users_after_filters):
    for item in users_after_filters:
        amount_of_authors = len(users_after_filters[item])
        train = users_after_filters[item][:int(0.8*amount_of_authors)]
        dev = users_after_filters[item][int(0.8*amount_of_authors):int(0.9*amount_of_authors)]
        test = users_after_filters[item][int(0.9*amount_of_authors):]
        write_to_directory(item,{'train':train,'test':test,'dev':dev})

def main():
    users_after_filters = analysis()
    split_train_test_dev(users_after_filters)
if __name__ == "__main__":
    main()
