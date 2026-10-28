"""Part A task 6: run the seven queries in sql/queries.sql with pd.read_sql_query and print the reported results."""
import re, sqlite3
import pandas as pd

text = open("sql/queries.sql", encoding="utf-8").read()
blocks = re.split(r"--\s*name:\s*(Q\d+)", text)[1:]
queries = {blocks[i]: blocks[i + 1].strip().rstrip(";") for i in range(0, len(blocks), 2)}

con = sqlite3.connect("data/instamart.db")
res = {name: pd.read_sql_query(sql, con) for name, sql in queries.items()}
con.close()

print("Q1 customers:", len(res["Q1"]))
q2 = res["Q2"]
print("Q2 rows:", len(q2), "| total revenue:", round(q2["revenue"].sum(), 2))
print("Q2 largest cell:", q2.iloc[0].to_dict())
print("Q3 products above average price:", len(res["Q3"]))
print("Q4 customers without completed order:", len(res["Q4"]))
q5 = res["Q5"]
print("Q5 customers above average:", len(q5), "| average spend:", q5["avg_total"].iloc[0])
q6 = res["Q6"]
print("Q6 top customer per city:")
print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows with RANK != ROW_NUMBER:", int((q6["rnk"] != q6["row_num"]).sum()))
q7 = res["Q7"]
print("Q7 daily rows:", len(q7), "| last cumulative:", q7["cumulative_revenue"].iloc[-1])
