import os
import sqlite3
import time
import warnings

import pandas as pd
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils import loadJsonFile
from text_sim import get_top_k_similar

warnings.filterwarnings('ignore')


a = "juierror/flan-t5-text2sql-with-schema"
b = "employee"
c = 512
d = 1024
e = 10
f = 10


def g(h):
    i = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(i, "..", "data", h)


def j(k, l):
    m = ", ".join(l)
    n = (
        "You are a database expert. Given the schema below, write a valid SQL query.\n\n"
        f"Schema: {m}\n\n"
        f"Task: {k}\n"
    )
    o = p(n, max_length=c, return_tensors="pt").input_ids
    return o


def q(k, l, r, s):
    m = ", ".join(l)
    t = (
        "You are a database expert. Given the schema and reference examples below, "
        "write a SQL query for the final task only.\n\n"
        f"Schema: {m}\n\n"
        "Reference Examples:\n"
    )
    for u, v in enumerate(r):
        t += f"Q: {v}\nSQL: {s[u]}\n\n"
    t += f"Now answer only this:\nQ: {k}\nSQL:"
    w = p(t, max_length=c, return_tensors="pt").input_ids
    return w


def x(y):
    y = y.to(z.device)
    aa = z.generate(inputs=y, num_beams=e, top_k=f, max_length=d)
    return p.decode(token_ids=aa[0], skip_special_tokens=True)


def ab(ac, ad):
    return ac.replace(" table", f" {ad}")


def ae(k, r, s, af):
    ag = get_top_k_similar(k, r, k=af)
    ah = [r[i] for i in ag]
    ai = [s[i] for i in ag]
    return ah, ai


def aj(ak, al):
    try:
        am = ak.execute(al).fetchall()
        print("Result:", am)
        return True
    except Exception as an:
        print(f"Execution failed: {an}")
        return False


def ao(ad):
    ap = g(f"example_queries/retr_set/final_{ad}.csv")
    aq = pd.read_csv(ap, delimiter="|")
    r = aq["Question"].tolist()
    s = aq["SQL Query"].tolist()
    return r, s


def ar(k, as_, ad, af=5):
    r, s = ao(ad)
    print(f"\nConnecting to database at: {as_}")
    ak = sqlite3.connect(as_)
    for _ in tqdm(range(2)):
        time.sleep(1)

    print(f"\nFinding top-{af} similar examples...")
    at, au = ae(k, r, s, af)

    for i, av in enumerate(at):
        print(f"  [{i+1}] Q: {av}")
        print(f"       SQL: {au[i]}")

    print("\n--- Zero-Shot Generation ---")
    aw = ab(x(j(k, ax)), ad)
    print("Query:", aw)
    ay = aj(ak, aw)
    print("Status:", "OK" if ay else "FAILED")

    print("\n--- Chain-of-Thought Generation ---")
    az = ab(x(q(k, ax, at, au)), ad)
    print("Query:", az)
    ay = aj(ak, az)
    print("Status:", "OK" if ay else "FAILED")

    ak.close()


def ba(as_, ad, af=5):
    bb = g(f"example_queries/test_set/final_{ad}.csv")
    bc = pd.read_csv(bb, delimiter="|")
    r, s = ao(ad)

    print(f"\nConnecting to database...")
    ak = sqlite3.connect(as_)
    for _ in tqdm(range(2)):
        time.sleep(1)

    for _, bd in bc.iterrows():
        k = bd["Question"]
        be = bd["SQL Query"]
        print(f"\n{'='*50}")
        print(f"Question : {k}")
        print(f"Expected : {be}")

        at, au = ae(k, r, s, af)

        print("\n[Zero-Shot]")
        aw = ab(x(j(k, ax)), ad)
        print("Generated:", aw)
        aj(ak, aw)

        print("\n[Chain-of-Thought]")
        az = ab(x(q(k, ax, at, au)), ad)
        print("Generated:", az)
        aj(ak, az)

    ak.close()


# ── Entry Point ──────────────────────────────────────────────────────────────

print("\nSQL Query Assistant")
print("-------------------")
bf = input("Enter target table name: ").strip()

bg = loadJsonFile(g("columns.json"), verbose=False)
ax = bg[bf]

print("\nLoading language model, please wait...")
p = AutoTokenizer.from_pretrained(a)
z = AutoModelForSeq2SeqLM.from_pretrained(a)

bh = input(
    "\nEnter your own question (y) or run full evaluation (n)? ").strip().lower()

bi = g("database/final_db.db")

if bh == "y":
    bj = input("Enter your question: ").strip()
    ar(bj, as_=bi, ad=bf)
else:
    ba(as_=bi, ad=bf)
