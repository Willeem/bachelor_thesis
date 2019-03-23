#!/usr/bin/python3

import praw
import pickle
from praw.models import MoreComments
from collections import defaultdict
from operator import itemgetter
import os
reddit = praw.Reddit(client_id='os.environ.get(CLIENT_ID)',
                     client_secret='os.environ.get(CLIENT_SECRET)',
                     user_agent='my user agent')

def get_top_1000(authors,timefilter):
    for submission in reddit.subreddit('soccer').top(timefilter,limit=1000):
        for comment in submission.comments:
            if isinstance(comment,MoreComments):
                continue
            if comment.author_flair_text:
                authors[comment.author_flair_text].add(comment.author)
    return authors
def get_usernames():
    authors = defaultdict(set)
    month_users = get_top_1000(authors,'month')
    extra_data = get_top_1000(month_users,'all')
    with open('clubs_with_authors.pickle','wb') as b:
        pickle.dump(extra_data,b,protocol = pickle.HIGHEST_PROTOCOL)


def clean_initial(authors):
    stripped_dict = defaultdict(set)
    for item in authors:
        stripped = item.strip(':')
        replaced = stripped.replace(' ','_')
        for kek in authors[item]:
            if replaced == "Barcelona":
                replaced = "FC_Barcelona"
            if replaced == "Bayern_Munich":
                replaced = "Bayern_München"
            stripped_dict[replaced].add(kek)
    return stripped_dict

def count_initial(stripped_dict):
    count_dict = defaultdict(int)
    for item in stripped_dict:
        count_dict[item] = len(stripped_dict[item])
    sorted_authors = sorted(count_dict.items(), key=itemgetter(1), reverse=True)
    return sorted_authors

def create_final_dict(sorted_authors,stripped_dict):
    total = 0
    final_dict = defaultdict(set)
    for item in sorted_authors:
        if item[1] > 100:
            final_dict[item[0]] = stripped_dict[item[0]]
            total += 1
    return final_dict

def create_folders(clubs_with_authors):
    for item in clubs_with_authors:
        outpath = os.getcwd() + "/data/" + str(item)
        os.mkdir(outpath)

def get_all_authors(final_dict):
    authors = [x.name for item in final_dict for x in final_dict[item]]
    return sorted(authors)


def main():
    #get_usernames()
    with open('clubs_with_authors.pickle','rb') as f:
        authors = pickle.load(f)
    stripped_dict = clean_initial(authors)
    sorted_authors = count_initial(stripped_dict)
    final_dict = create_final_dict(sorted_authors,stripped_dict)
    all_authors_sorted = get_all_authors(final_dict)
    with open('final_dict.pickle','wb') as b:
        pickle.dump(final_dict,b, protocol = pickle.HIGHEST_PROTOCOL)
    with open('authors.pickle','wb') as b:
        pickle.dump(all_authors_sorted,b, protocol = pickle.HIGHEST_PROTOCOL)

    create_folders(final_dict)

if __name__ == "__main__":
    main()
