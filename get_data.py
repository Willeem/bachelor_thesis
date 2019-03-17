#!/usr/bin/python3

import praw
import pickle
from praw.models import MoreComments
from collections import defaultdict
reddit = praw.Reddit(client_id='YpYx4r89UxeMag',
                     client_secret='7nLiowFm1BmpNxmHlvirrEe8Hj8',
                     user_agent='my user agent')

authors = defaultdict(set)
for submission in reddit.subreddit('soccer').top('month',limit=1000):
    for comment in submission.comments:
        if isinstance(comment,MoreComments):
            continue
        if comment.author_flair_text:
            authors[comment.author_flair_text].add(comment.author)

with open('clubs_with_authors.pickle','wb') as b:
    pickle.dump(authors,b, protocol = pickle.HIGHEST_PROTOCOL)
