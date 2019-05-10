#!/usr/bin/python3

from os import listdir, makedirs, getcwd
from os.path import isdir, isfile, join
from collections import defaultdict
from shutil import copy2
from operator import itemgetter


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

def print_analysis(analysis_dict):
    sorted_analysis = sorted(analysis_dict.items(), key=itemgetter(0))
    total_users = 0
    total_words = 0
    print("\t%-25s\t%-5s\t %.5s \t%.5s" % ("Club", "Users", "Words", "Average words per user"))
    for i in range(len(sorted_analysis)):
        item = sorted_analysis[i][0]
        if sorted_analysis[i][1]['users'] > 0:
            total_users += analysis_dict[item]['users']
            total_words += analysis_dict[item]['words']
            print("\t%-25s & %-5i & %-5i & %.2f \\\\" % (item, analysis_dict[item]['users'],analysis_dict[item]['words'], (analysis_dict[item]['words']/analysis_dict[item]['users'])))
    print("\t%-25s & %-5i & %-5i & %.2f \\\\" % ("Total", total_users, total_words,(total_words/total_users)))


def analysis():
    teams = get_items_in_directory('directory','data/')
    users_after_filters = defaultdict(list)
    analysis_dict = {}
    total_users = 0
    for team in teams:
        analysis_dict[team] = defaultdict(int)
        if team not in ['Argentina','Australia','Belgium','Brazil','Colombia',
        'Croatia','England','France','Germany','Mexico','Poland','Portugal',
        'Republic_of_Ireland','Sweden','The_Netherlands','United_States']:
            users = 0
            word_count = 0
            files = get_items_in_directory('file','data/' + team)
            for user in files:
                data = open('data/' + team + '/' + user, 'r', encoding='utf-8').read()
                split_data = data.split('##########')
                word_count = count_words(split_data)
                if word_count > 2000:
                    analysis_dict[team]['words'] += word_count
                    users += 1
                    users_after_filters[team].append(user)
            if users < 50:
                del users_after_filters[team]
            else:
                total_users += users
                analysis_dict[team]['users'] += users

    print("Total users: {} from {} teams.".format(total_users,len(users_after_filters.keys())))
    print_analysis(analysis_dict)
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
