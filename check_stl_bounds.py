import sys, struct, re
from pathlib import Path

def read_stl_vertices(path):
    data = Path(path).read_bytes()

    # Binary STL: 80 byte header + uint32 tri count + 50 bytes per triangle
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            verts = []
            off = 84
            for _ in range(n):
                off += 12  # normal
                for _ in range(3):
                    verts.append(struct.unpack("<fff", data[off:off+12]))
                    off += 12
                off += 2
            return verts

    # ASCII STL fallback
    text = data.decode(errors="ignore")
    verts = []
    for line in text.splitlines():
        m = re.search(r"\bvertex\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)", line)
        if m:
            verts.append(tuple(float(x) for x in m.groups()))
    return verts

for p in sys.argv[1:]:
    verts = read_stl_vertices(p)
    if not verts:
        print(p, "NO VERTICES")
        continue
    xs, ys, zs = zip(*verts)
    mn = (min(xs), min(ys), min(zs))
    mx = (max(xs), max(ys), max(zs))
    cen = tuple((a+b)/2 for a,b in zip(mn,mx))
    size = tuple(b-a for a,b in zip(mn,mx))
    print("\n", p)
    print("  min   ", mn)
    print("  max   ", mx)
    print("  center", cen)
    print("  size  ", size)
