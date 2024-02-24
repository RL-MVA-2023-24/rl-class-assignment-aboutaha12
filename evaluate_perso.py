normal_scores = [3432807, 1e8, 1e9, 1e10, 2e10, 5e10]
scores_dr = [5e10, 1e10, 5e10]

with open("score.txt", "r") as f:
    lines = f.read().splitlines()

score = float(lines[0])
score_dr = float(lines[1])

s = 0
l = []
for k in normal_scores:
    l.append(k < score)
    s += k < score

for k in scores_dr:
    l.append(k < score_dr)
    s += k < score_dr

print(score, score_dr, l, s)
