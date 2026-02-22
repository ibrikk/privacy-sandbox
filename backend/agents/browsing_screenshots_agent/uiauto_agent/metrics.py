from collections import Counter
import math

def app_distribution(history):
    apps = [h["action"]["app"] for h in history if "action" in h]
    counts = Counter(apps)
    total = sum(counts.values())

    return {app: c / total for app, c in counts.items()}

def entropy(distribution):
    return -sum(p * math.log(p + 1e-8) for p in distribution.values())

def repetition_rate(history):
    repeats = 0
    for i in range(1, len(history)):
        if history[i]["action"]["app"] == history[i-1]["action"]["app"]:
            repeats += 1
    return repeats / max(1, len(history))