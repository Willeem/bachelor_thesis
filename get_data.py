#!/usr/bin/python3

import praw
import pickle
from praw.models import MoreComments
from collections import defaultdict
from operator import itemgetter
import os
from decouple import config


reddit = praw.Reddit(client_id=config('CLIENT_ID'),
                     client_secret=config('CLIENT_SECRET'),
                     user_agent=config('USER_ID'))


def get_top_1000(authors, timefilter):
    """Retrieves the top 1000 posts of the soccer subreddit from a given timefilter """
    for submission in reddit.subreddit('soccer').top(timefilter, limit=1000):
        for comment in submission.comments:
            if isinstance(comment, MoreComments):
                continue
            if comment.author_flair_text:
                authors[comment.author_flair_text].add(comment.author)
    return authors


def get_usernames():
    """Gets the usernames needed and writes them to a pickle file"""
    authors = defaultdict(set)
    month_users = get_top_1000(authors, 'month')
    extra_data = get_top_1000(month_users, 'all')
    with open('clubs_with_authors.pickle', 'wb') as b:
        pickle.dump(extra_data, b, protocol=pickle.HIGHEST_PROTOCOL)


def clean_initial(authors):
    """Cleans the name of the Reddit flairs and
    combines some different spellings of a team into one class"""
    stripped_dict = defaultdict(set)
    for item in authors:
        stripped = item.strip(':')
        replaced = stripped.replace(' ', '_')
        for kek in authors[item]:
            if replaced == "Barcelona":
                replaced = "FC_Barcelona"
            if replaced == "Bayern_Munich":
                replaced = "Bayern_München"
            stripped_dict[replaced].add(kek)
    return stripped_dict


def count_initial(stripped_dict):
    """Counts and sorts the amount of users per team"""
    count_dict = defaultdict(int)
    total_count = 0
    for item in stripped_dict:
        count_dict[item] = len(stripped_dict[item])
        total_count += len(stripped_dict[item])
    sorted_authors = sorted(
        count_dict.items(), key=itemgetter(1), reverse=True)
    print("Total count: {} from {} teams.".format(
        total_count, len(count_dict.keys())))
    return sorted_authors


def create_final_dict(sorted_authors, stripped_dict):
    """Filters the teams represented by less than 100 users"""
    total = 0
    final_dict = defaultdict(set)
    for item in sorted_authors:
        if item[1] > 100:
            final_dict[item[0]] = stripped_dict[item[0]]
            total += item[1]
    print("Usable users: {} from {} teams".format(
        total, len(final_dict.keys())))
    return final_dict


def create_folders(clubs_with_authors):
    """Creates a folder per team"""
    for item in clubs_with_authors:
        outpath = os.getcwd() + "/data/" + str(item)
        os.mkdir(outpath)


def get_all_authors(final_dict):
    """Retrieves and sorts all authors"""
    authors = [x.name for item in final_dict for x in final_dict[item]]
    return sorted(authors)


def main():
    authors = get_usernames()
    stripped_dict = clean_initial(authors)
    sorted_authors = count_initial(stripped_dict)
    print(sorted_authors)
    final_dict = create_final_dict(sorted_authors, stripped_dict)
    all_authors_sorted = get_all_authors(final_dict)
    with open('final_dict.pickle', 'wb') as b:
        pickle.dump(final_dict, b, protocol=pickle.HIGHEST_PROTOCOL)
    with open('authors.pickle', 'wb') as b:
        pickle.dump(all_authors_sorted, b, protocol=pickle.HIGHEST_PROTOCOL)

    create_folders(final_dict)


if __name__ == "__main__":
    main()
