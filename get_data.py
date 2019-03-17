#!/usr/bin/python3

import praw
import pickle
from praw.models import MoreComments
from collections import defaultdict
reddit = praw.Reddit(client_id='os.environ.get(CLIENT_ID)',
                     client_secret='os.environ.get(CLIENT_SECRET)',
                     user_agent='my user agent')

authors = defaultdict(set)
for submission in reddit.subreddit('soccer').top('month',limit=1000):
    for comment in submission.comments:
        if isinstance(comment,MoreComments):
            continue
        if comment.author_flair_text:
            authors[comment.author_flair_text].add(comment.author)

with open('clubs_with_authors.pickle','w') as b:
    b.dump(authors)
