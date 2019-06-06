#!/usr/bin/python3
import praw
import prawcore
import os
import pickle
import logging
from datetime import datetime

reddit = praw.Reddit(client_id=os.environ.get('CLIENT_ID'),
                     client_secret=os.environ.get('CLIENT_SECRET'),
                     user_agent=os.environ.get('USER_ID'))

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up as many loggers as you want"""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


user_logger = setup_logger('user_logger', 'log/users.log')
with open('authors.pickle', 'rb') as f:
    authors = pickle.load(f)

with open('final_dict.pickle', 'rb') as f:
    clubs_with_authors = pickle.load(f)


def make_searchable_dict(clubs_with_authors):
    """Creates a dictionary with the username as key and their favourite team as value"""
    return {x.name: item for item in clubs_with_authors for x in clubs_with_authors[item]}


def get_comments(author, author_with_club):
    """Retrieves comments from the soccer subreddit from the Reddit API and
    writes them to the correct directory"""
    club = author_with_club[author]
    count = 0
    path_name = os.getcwd() + "/data/" + club + "/" + author + ".txt"
    with open(path_name, 'w') as f:
        for comment in reddit.redditor(str(author)).comments.new(limit=None):
            if comment.subreddit_id == "t5_2qi58":
                f.write("{}\n ########## \n".format(comment.body))
                count += 1
    user_logger.info("Wrote {} comments to file {}".format(count, path_name))


def main():
    startTime = datetime.now()
    author_with_club = make_searchable_dict(clubs_with_authors)
    for author in authors:
        try:
            get_comments(author, author_with_club)
        except:
            user_logger.error("Could not find the user profile of {} with club {}".format(
                author, author_with_club[author]))
    user_logger.info("Took {}".format(datetime.now() - startTime))


if __name__ == "__main__":
    main()
