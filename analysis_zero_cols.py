import csv, collections

rows = list(csv.DictReader(open(r'Z:\codeSpace\AMLS\ge_blocks.csv')))

sep_bytes = 0.0
sep_zero_bytes = 0.0
by_level = collections.defaultdict(lambda: [0.0, 0.0])
leaf_bytes = 0.0
leaf_zero = 0.0
for r in rows:
    n_a = int(r['n_a'])
    n_i = int(r['n_i'])
    b = n_a * n_i
    z = b * (1 - float(r['kFrac']))
    if r['leaf'] == '1':
        leaf_bytes += b
        leaf_zero += z
    else:
        sep_bytes += b
        sep_zero_bytes += z
        lv = min(int(r['level']), 7)
        by_level[lv][0] += b
        by_level[lv][1] += z

tot = sep_bytes + leaf_bytes
print("总稠密字节(全体列块): %.2f GB" % (tot / 1e9))
print("  sep 块: %.2f GB, 零列 %.2f GB (%.1f%%)" % (sep_bytes / 1e9, sep_zero_bytes / 1e9, 100 * sep_zero_bytes / sep_bytes))
print("  叶块: %.2f GB, 零列 %.2f GB (%.1f%%)" % (leaf_bytes / 1e9, leaf_zero / 1e9, 100 * leaf_zero / leaf_bytes))
print("  压缩 sep 块的真实收益(相对全体): %.1f%%" % (100 * sep_zero_bytes / tot))
print()
for lv in sorted(by_level):
    b, z = by_level[lv]
    print("  level%d: %.3f GB, 零列占比 %.1f%%" % (lv, b / 1e9, 100 * z / b))
