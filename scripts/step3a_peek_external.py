"""
OttoLatin - Adim 3a: Dis test setlerinin icine bak
Kullanim:
    python scripts/step3a_peek_external.py
    (data/external/raw/ icindeki .zip / .tar.gz dosyalarini data/external/extracted/ altina acar)
Cikti: out/step3/peek.txt  -> bunu Claude'a yukle
"""
import os
import tarfile
import zipfile

RAW = "data/external/raw"
EXT = "data/external/extracted"
OUT = "out/step3/peek.txt"
TEXT_EXT = {".txt", ".csv", ".tsv", ".json", ".jsonl", ".xml", ".md", ".gt", ".html"}

os.makedirs(RAW, exist_ok=True)
if not [f for f in os.listdir(RAW) if f.endswith((".zip", ".tar.gz", ".tgz", ".tar"))]:
    raise SystemExit(f"[!] {os.path.abspath(RAW)} klasoru bos. Indirdigin .zip / .tar.gz dosyalarini buraya koyup tekrar calistir.")
os.makedirs(EXT, exist_ok=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
lines = []
p = lines.append


def safe_members_tar(tf, dest):
    for m in tf.getmembers():
        target = os.path.realpath(os.path.join(dest, m.name))
        if target.startswith(os.path.realpath(dest)) and (m.isfile() or m.isdir()):
            yield m


for fn in sorted(os.listdir(RAW)):
    src = os.path.join(RAW, fn)
    dest = os.path.join(EXT, fn.split(".")[0])
    os.makedirs(dest, exist_ok=True)
    try:
        if fn.endswith(".zip"):
            with zipfile.ZipFile(src) as z:
                for n in z.namelist():
                    t = os.path.realpath(os.path.join(dest, n))
                    if t.startswith(os.path.realpath(dest)):
                        z.extract(n, dest)
        elif fn.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(src) as tf:
                tf.extractall(dest, members=safe_members_tar(tf, dest), filter="data")
        else:
            p(f"[atlandi] {fn}")
            continue
        p(f"[acildi] {fn} -> {dest}")
    except Exception as e:
        p(f"[HATA] {fn}: {e}")

p("\n================ DOSYA LISTESI ================")
all_files = []
for root, _, files in os.walk(EXT):
    for f in files:
        path = os.path.join(root, f)
        all_files.append(path)
        p(f"{os.path.getsize(path):>10,} B  {path}")

p("\n================ ICERIK ORNEKLERI ================")
by_dir_count = {}
for path in sorted(all_files):
    ext = os.path.splitext(path)[1].lower()
    d = os.path.dirname(path)
    by_dir_count[d] = by_dir_count.get(d, 0) + 1
    if by_dir_count[d] > 4:          # her klasorden en fazla 4 dosya goster
        continue
    if ext in TEXT_EXT or ext == "":
        p(f"\n----- {path} -----")
        try:
            with open(path, encoding="utf-8-sig", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= 15:
                        p("...")
                        break
                    p(line.rstrip("\n")[:300])
        except Exception as e:
            p(f"[okunamadi] {e}")
    elif ext in {".xlsx", ".xls"}:
        p(f"\n----- {path} (excel) -----")
        try:
            import pandas as pd
            for name, sh in pd.read_excel(path, sheet_name=None).items():
                p(f"[sayfa: {name}] {sh.shape}  sutunlar: {list(sh.columns)}")
                p(sh.head(8).to_string()[:2000])
        except Exception as e:
            p(f"[okunamadi] {e}  (pip install openpyxl gerekebilir)")
    else:
        p(f"[ikili/goruntu] {path}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines[:40]))
print(f"\n[i] Tam rapor: {OUT}")
