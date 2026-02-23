import datetime

def _timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def section(title):
    #print("\n" + "-" * 70)
    print(f"[{_timestamp()}] {title}")
    #print("-" * 70)

def info(message):
    print(f"    ... {message}")

def warn(message):
    print(f"   WARNING: {message}")