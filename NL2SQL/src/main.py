import os
import sqlite3
import time
import warnings

import pandas as pd
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils import dumpJsonFile, loadJsonFile
from text_sim import get_top_k_similar

warnings.filterwarnings('ignore')


CHECKPOINT = "juierror/flan-t5-text2sql-with-schema"
DEFAULT_TABLE = "employee"
MAX_INPUT_LEN = 512
MAX_OUTPUT_LEN = 1024
BEAM_COUNT = 10
TOP_K_BEAMS = 10


def get_data_path(relative_path: str) -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "..", "data", relative_path)


def build_prompt(user_query: str, schema_fields: List[str]) -> object:
    schema_str = ", ".join(schema_fields)
    prompt = (
        "You are a database expert. Given the schema below, write a valid SQL query.\n\n"
        f"Schema: {schema_str}\n\n"
        f"Task: {user_query}\n"
    )
    token_ids = tokenizer(prompt, max_length=MAX_INPUT_LEN,
                          return_tensors="pt").input_ids
    return token_ids


def build_cot_prompt(
    user_query: str,
    schema_fields: List[str],
    ref_questions: List[str],
    ref_queries: List[str]
) -> object:
    schema_str = ", ".join(schema_fields)
    prompt = (
        "You are a database expert. Given the schema and reference examples below, "
        "write a SQL query for the final task only.\n\n"
        f"Schema: {schema_str}\n\n"
        "Reference Examples:\n"
    )
    for idx, q in enumerate(ref_questions):
        prompt += f"Q: {q}\nSQL: {ref_queries[idx]}\n\n"

    prompt += f"Now answer only this:\nQ: {user_query}\nSQL:"

    token_ids = tokenizer(prompt, max_length=MAX_INPUT_LEN,
                          return_tensors="pt").input_ids
    return token_ids


def run_model(token_ids) -> str:
    token_ids = token_ids.to(model.device)
    output_ids = model.generate(
        inputs=token_ids,
        num_beams=BEAM_COUNT,
        top_k=TOP_K_BEAMS,
        max_length=MAX_OUTPUT_LEN
    )
    return tokenizer.decode(token_ids=output_ids[0], skip_special_tokens=True)


def fix_table_reference(query: str, tbl: str) -> str:
    return query.replace(" table", f" {tbl}")


def fetch_similar_examples(user_query: str, ref_questions: List[str], ref_queries: List[str], k: int):
    ranked_indices = get_top_k_similar(user_query, ref_questions, k=k)
    matched_q = [ref_questions[i] for i in ranked_indices]
    matched_sql = [ref_queries[i] for i in ranked_indices]
    return matched_q, matched_sql


def execute_query(conn: sqlite3.Connection, sql: str):
    try:
        rows = conn.execute(sql).fetchall()
        print("Result:", rows)
        return True
    except Exception as e:
        print(f"Execution failed: {e}")
        return False


def load_reference_data(tbl: str):
    retr_path = get_data_path(f"example_queries/retr_set/final_{tbl}.csv")
    retr_df = pd.read_csv(retr_path, delimiter="|")
    questions = retr_df["Question"].tolist()
    queries = retr_df["SQL Query"].tolist()
    return questions, queries


def run_single(user_query: str, db_path: str, tbl: str = DEFAULT_TABLE, k: int = 5):
    ref_questions, ref_queries = load_reference_data(tbl)

    print(f"\nConnecting to database at: {db_path}")
    conn = sqlite3.connect(db_path)
    for _ in tqdm(range(2)):
        time.sleep(1)

    print(
        f"\nFinding top-{k} similar examples for chain-of-thought context...")
    sim_questions, sim_queries = fetch_similar_examples(
        user_query, ref_questions, ref_queries, k)

    for i, sq in enumerate(sim_questions):
        print(f"  [{i+1}] Q: {sq}")
        print(f"       SQL: {sim_queries[i]}")

    print("\n--- Zero-Shot Generation ---")
    zs_sql = fix_table_reference(
        run_model(build_prompt(user_query, schema)), tbl)
    print("Query:", zs_sql)
    success = execute_query(conn, zs_sql)
    print("Status:", "OK" if success else "FAILED")

    print("\n--- Chain-of-Thought Generation ---")
    cot_sql = fix_table_reference(
        run_model(build_cot_prompt(user_query, schema,
                  sim_questions, sim_queries)), tbl
    )
    print("Query:", cot_sql)
    success = execute_query(conn, cot_sql)
    print("Status:", "OK" if success else "FAILED")

    conn.close()


def run_evaluation(db_path: str, tbl: str = DEFAULT_TABLE, k: int = 5):
    test_path = get_data_path(f"example_queries/test_set/final_{tbl}.csv")
    test_df = pd.read_csv(test_path, delimiter="|")
    ref_questions, ref_queries = load_reference_data(tbl)

    print(f"\nConnecting to database...")
    conn = sqlite3.connect(db_path)
    for _ in tqdm(range(2)):
        time.sleep(1)

    for _, row in test_df.iterrows():
        q = row["Question"]
        ground_truth = row["SQL Query"]
        print(f"\n{'='*50}")
        print(f"Question : {q}")
        print(f"Expected : {ground_truth}")

        sim_questions, sim_queries = fetch_similar_examples(
            q, ref_questions, ref_queries, k)

        print("\n[Zero-Shot]")
        zs_sql = fix_table_reference(run_model(build_prompt(q, schema)), tbl)
        print("Generated:", zs_sql)
        execute_query(conn, zs_sql)

        print("\n[Chain-of-Thought]")
        cot_sql = fix_table_reference(
            run_model(build_cot_prompt(
                q, schema, sim_questions, sim_queries)), tbl
        )
        print("Generated:", cot_sql)
        execute_query(conn, cot_sql)

    conn.close()


# ── Entry Point ──────────────────────────────────────────────────────────────

print("\nSQL Query Assistant")
print("-------------------")
tbl_name = input("Enter target table name: ").strip()

column_data = loadJsonFile(get_data_path("columns.json"), verbose=False)
schema = column_data[tbl_name]

print("\nLoading language model, please wait...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)

mode = input(
    "\nEnter your own question (y) or run full evaluation (n)? ").strip().lower()

db_file = get_data_path("database/final_db.db")

if mode == "y":
    user_q = input("Enter your question: ").strip()
    run_single(user_q, db_path=db_file, tbl=tbl_name)
else:
    run_evaluation(db_path=db_file, tbl=tbl_name)
