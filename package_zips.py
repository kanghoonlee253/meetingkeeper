"""Packs dist/MeetingKeeper into MeetingKeeper_1.zip … MeetingKeeper_N.zip, each under 2 GB.

Files bigger than PART_SIZE are stored as <name>.part1ofK …; the app joins them on first start.
Every zip contains paths under MeetingKeeper/, so extracting all of them to the same place
rebuilds one folder.
"""
import os
import sys
import zipfile

DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist")
SRC = "MeetingKeeper"
LIMIT = 1_900_000_000        # bytes of content per zip (safely below 2 GB)
PART_SIZE = 1_000_000_000


def items():
    """(archive_name, source_path, offset, length) for every file / file part."""
    out = []
    for root, _, files in os.walk(os.path.join(DIST, SRC)):
        for f in files:
            path = os.path.join(root, f)
            arc = os.path.relpath(path, DIST)
            size = os.path.getsize(path)
            if size <= PART_SIZE:
                out.append((arc, path, 0, size))
                continue
            n = -(-size // PART_SIZE)
            for i in range(n):
                off = i * PART_SIZE
                out.append((f"{arc}.part{i + 1}of{n}", path, off, min(PART_SIZE, size - off)))
    return out


def main():
    os.chdir(DIST)
    for old in os.listdir("."):
        if old.startswith("MeetingKeeper") and old.endswith(".zip"):
            os.remove(old)
    # Zip 1 holds the whole program (exe, _internal, guide) so a missing later zip can only ever mean
    # missing model files, which the app reports. Models fill the rest, largest first.
    is_model = lambda it: it[0].startswith(os.path.join(SRC, "models") + os.sep)
    program = [it for it in items() if not is_model(it)]
    big = sorted((it for it in items() if is_model(it)), key=lambda it: -it[3])
    bins = [[program, sum(it[3] for it in program)]]
    for it in big:
        for b in bins:
            if b[1] + it[3] <= LIMIT:
                b[0].append(it)
                b[1] += it[3]
                break
        else:
            bins.append([[it], it[3]])
    for i, (contents, _) in enumerate(bins, 1):
        name = f"MeetingKeeper_{i}.zip"
        with zipfile.ZipFile(name, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for arc, path, off, length in contents:
                if off == 0 and length == os.path.getsize(path):
                    z.write(path, arc)
                    continue
                with open(path, "rb") as src, z.open(zipfile.ZipInfo(arc.replace(os.sep, "/")), "w",
                                                     force_zip64=True) as dst:
                    src.seek(off)
                    left = length
                    while left:
                        chunk = src.read(min(16 << 20, left))
                        dst.write(chunk)
                        left -= len(chunk)
        size = os.path.getsize(name)
        print(f"{name}: {size / 2**30:.2f} GiB ({size:,} bytes), {len(contents)} files")
        if size >= 2_000_000_000:
            sys.exit(f"{name} is over 2 GB")


if __name__ == "__main__":
    main()
