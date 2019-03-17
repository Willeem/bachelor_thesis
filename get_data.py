#!/usr/bin/python3

import praw
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
print(len(authors))
for item in authors:
    print(item,len(authors[item]))
