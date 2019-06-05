#!/usr/bin/python3
import pickle

f=open('pos_tags_train.pickle', 'rb')
f2=open('pos_tags_dev.pickle', 'rb')
f3=open('pos_tags_test.pickle', 'rb')
objs = []
while 1:
    try:
        objs.append(pickle.load(f))
        objs.append(pickle.load(f2))
        objs.append(pickle.load(f3))
    except EOFError:
        break

count = 0
for obj in objs:
    print(obj)
    count += len(obj)
print(count)
