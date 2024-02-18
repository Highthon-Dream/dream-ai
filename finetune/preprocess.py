with open("datas.txt", "r", encoding="utf-8") as f:
    content = f.read()

with open("result.txt", "w", encoding="utf-8") as f:
    a = [f"\"{' '.join(row.split('	')[1:]).strip()}\"" for row in content.split("\n")]
    f.write(",".join(a))