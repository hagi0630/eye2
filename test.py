from collections import deque

n,q = map(int,input().split())
s = list(input())



cnt = 0
for _ in range(q):
    t,x = map(int,input().split())
    if t==1:
        cnt+=x
    else:
        ans_index = (cnt+x)%n
        print(s[ans_index])