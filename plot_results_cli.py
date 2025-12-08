import os

RESULT_DIR = "results"

def parse_log(path):
    stats = {}
    with open(path) as f:
        for line in f:
            line = line.strip()

            if line.startswith("Total cycles:"):
                stats["cycles"] = int(line.split()[-1])

            elif line.startswith("Total instructions:"):
                stats["instr"] = int(line.split()[-1])

            elif line.startswith("IPC:"):
                stats["ipc"] = float(line.split()[-1])

            elif line.startswith("L1 hits:"):
                nums = [int(s) for s in line.replace(",", "").split() if s.isdigit()]
                stats["l1_hits"] = nums[0]
                stats["l1_miss"] = nums[1]

            elif line.startswith("L2 hits:"):
                nums = [int(s) for s in line.replace(",", "").split() if s.isdigit()]
                stats["l2_hits"] = nums[0]
                stats["l2_miss"] = nums[1]



            elif line.startswith("DRAM accesses:"):
                stats["dram"] = int(line.split()[-1])

    return stats

def ascii_bar(value, max_value, width=40):
    filled = int((value / max_value) * width)
    return "#" * filled + "-" * (width - filled)

logs = [f for f in os.listdir(RESULT_DIR) if f.endswith(".txt")]
logs.sort()

print("Found results:")
for log in logs:
    print("  ", log)
print("\n====================\n")

all_stats = []
for log in logs:
    path = os.path.join(RESULT_DIR, log)
    stats = parse_log(path)
    stats["file"] = log
    all_stats.append(stats)

max_ipc = max(s["ipc"] for s in all_stats)

print("IPC COMPARISON (ASCII BAR CHART)")
print("--------------------------------")
for s in all_stats:
    bar = ascii_bar(s["ipc"], max_ipc)
    print(f"{s['file']:<30} IPC={s['ipc']:.4f}  {bar}")

print("\nDETAIL TABLE")
print("-----------------------------------------------")
print(f"{'file':30s} {'IPC':>8} {'Cycles':>10} {'Instr':>10} {'L1_hit%':>10} {'L2_hit%':>10}")

for s in all_stats:
    l1_rate = s["l1_hits"] / (s["l1_hits"] + s["l1_miss"]) * 100
    l2_rate = s["l2_hits"] / (s["l2_hits"] + s["l2_miss"]) * 100
    print(f"{s['file']:30s} "
          f"{s['ipc']:8.4f} "
          f"{s['cycles']:10d} "
          f"{s['instr']:10d} "
          f"{l1_rate:9.2f}% "
          f"{l2_rate:9.2f}%")
