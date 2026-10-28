# Instamart Customer Intelligence & Support Copilot

## Overview

One repository, four connected Parts, all on synthetic and fictional data (all money in INR):

| Part | What it does | Status |
|---|---|---|
| A | Clean the raw exports, store them in SQLite, answer business questions in SQL and pandas | done |
| B | RFM segments (K-means) and a lapse-prediction model | done |
| C | Support-ticket triage with privacy and safety guardrails | done |
| D | Policy RAG assistant and a refund agent | done |

`generate_data.py` and `mock_llm.py` are the supplied files and are committed unmodified. No API key, network call or paid service is used anywhere.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate_data.py          # last line: generated: customers=1200 products=40 orders_raw=13084 tickets=164
python part_a_audit.py           # raw-data audit figures
python part_a_clean.py           # cleaning rules -> data/clean/*.csv and data/instamart.db
python validate.py               # must print ALL CHECKPOINTS PASSED
python run_queries.py            # runs the seven queries in sql/queries.sql
python part_a_analysis.py        # pandas cross-check and exploratory statistics
python part_b_segments.py        # RFM, K-means sweep, segments, PCA -> outputs/customer_segments.csv
python part_b_lapse.py           # lapse model -> outputs/lapse_model.joblib, outputs/retention_list.csv
python part_c_triage.py          # triage -> outputs/triage_results.csv, outputs/triage_budget_demo.csv
python part_d_retrieval.py       # chunking, retrieval, policy evaluation (overlap 10 and 0)
python part_d_agent.py           # refund agent -> outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl, state/
python -m pytest -q              # 44 automated tests (run after the commands above)
```

Run these from the repository root, in this order. `part_d_agent.py` deletes and recreates the ledger, run log and `state/` first, so repeated runs give identical results. The tests check the artifacts the earlier commands produce, so run them last.

## Part A

### Audit of the raw data

| Table | Shape | Notes |
|---|---|---|
| customers.csv | 1200 x 5 | age 61 missing (5.08%), gender 30 missing (2.50%) |
| products.csv | 40 x 4 | no missing values |
| orders.csv | 13084 x 9 | delivery_minutes 2541 blank (19.42%); order_amount holds text such as N/A and error |

| Audit figure | Value |
|---|---|
| raw orders rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount entries | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| blank age | 61 |
| age outside 10-100 | 6 |
| distinct raw city spellings | 24 |
| distinct non-blank gender spellings | 10 |
| blank gender | 30 |

Note: `orders.csv` is read with `keep_default_na=False, na_values=[""]`. Otherwise pandas silently turns the text `N/A` into a missing value and the non-numeric count would show 67 instead of 120.

### Cleaning rules and what each changed

Customers
1. City: strip spaces and apply title case, so the 24 raw spellings become 6 cities (Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184).
2. Gender: every spelling mapped to male, female or unknown; the 30 blanks become unknown (female 610, male 560, unknown 30).
3. Age: the 6 values outside 10-100 set to missing, then all 67 missing ages (61 blank + 6) filled with the median age of the customer's city using `groupby().transform("median")`. Result: 0 missing, all ages between 18 and 60.
4. signup_date converted to datetime.

Orders (applied in this order)
1. order_ts converted to datetime.
2. Exact duplicate rows removed: 119 rows.
3. For each order_id only the record with the latest order_ts is kept: 656 older pending records removed. Result: 12,309 rows, one per order_id.
4. order_amount converted to numeric (unparseable becomes missing); 112 missing amounts filled with `round(price_inr x quantity x (1 - discount_pct/100), 2)`. (The raw 120 unparseable entries shrink to fewer after de-duplication.)
5. delivery_minutes: 105 negative values set to missing; then 774 completed orders with missing delivery time filled with the median of their product category among completed orders (`groupby().transform`). Cancelled orders stay missing because a cancelled order was never delivered.

After cleaning: 11,110 completed and 1,199 cancelled orders; no pending rows; 0 missing order_amount; 0 missing delivery_minutes for completed orders and 1,199 for cancelled ones; completed revenue INR 2,998,668.80. `validate.py` asserts 15 checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL results (sql/queries.sql, run against data/instamart.db)

| Query | Result |
|---|---|
| Q1 WHERE then HAVING | 30 customers |
| Q2 three-table join | 30 rows; revenues add up to INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products priced above the catalogue average |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 CTE | 550 customers above the average spend of INR 2,552.06 |
| Q6 window ranks | top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK() different from ROW_NUMBER() |
| Q7 running total | 183 daily rows; last cumulative value INR 2,998,668.80 |

Why ties cause the RANK()/ROW_NUMBER() difference: when two customers in a city have exactly the same total spend, RANK() gives them the same rank (and skips the next number), while ROW_NUMBER() still assigns each a different consecutive number.

### pandas cross-check and exploratory analysis

The pivot table with `margins=True` gives a grand total of INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed order_amount: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, fences -283.50 and 645.30, with 0 outliers below and 897 above.

Correlation matrix (completed orders): order_amount vs discount_pct -0.095 (largest in absolute value), order_amount vs quantity +0.087, order_amount vs delivery_minutes -0.003, discount_pct vs delivery_minutes -0.010, quantity vs delivery_minutes -0.013, discount_pct vs quantity 0.000.

Four questions, in under 200 words:
- **What is this data about?** Six months (April to September 2025) of quick-commerce orders in six Indian cities, with customers and a 40-product catalogue.
- **Distributions?** Order amounts are strongly right-skewed: the mean (269.91) is about double the median (135.00), because most baskets are small groceries and a few are electronics.
- **Outliers?** 897 orders exceed the upper fence of 645.30 and none fall below. They are mostly genuine electronics orders, not errors, so they are kept.
- **Relationships?** The largest correlation is order_amount with discount_pct at about -0.095, then quantity at +0.087. Both are very weak, and a correlation of this size does not establish causation.

## Part B

Scripts: `part_b_segments.py` (tasks 1-3) and `part_b_lapse.py` (tasks 4-8). Seeds are fixed at 42 everywhere.

**Leakage statement.** No encoder, scaler or resampler ever saw test rows (all preprocessing sits inside a scikit-learn `Pipeline` fitted on training rows only, and the over-sampler is applied to the training rows only). No order dated 2025-09-01 or later entered the lapse features. The segment labels from task 3 were not used as features.

### RFM table (snapshot 2025-10-01 00:00:00, completed orders only)

1,175 customers. Recency: median 19, min 0, max 180 days. Frequency: mean 9.46, max 33. Monetary: mean INR 2,552.06, max INR 14,766.55.

### K-means sweep (StandardScaler, `n_init=10`, `random_state=42`)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Choice: K = 4.** The elbow is at K=4 (inertia drops by 525 from K=3 to K=4, but only by 126 from K=4 to K=5), K=4 has the best silhouette (0.526), and four groups are a number the business can act on with four different campaigns. Inertia of K=8 is always lower than K=2 because more centres can only shorten the distances from points to their nearest centre (at K = number of customers it would be zero), so a lower inertia alone cannot choose K; it has to be read together with the elbow, the silhouette and interpretability.

### Segment profile (K = 4)

| Segment | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|
| Premium Big Spenders (premium) | 138 | 24.1 | 6.77 | 7,129.70 | VIP perks, early access to electronics deals, personal retention outreach |
| Loyal Regulars | 419 | 8.8 | 19.40 | 3,410.02 | Loyalty points and basket bundles to protect their habit |
| Occasional Shoppers | 436 | 28.9 | 3.77 | 1,057.62 | Frequency nudges, small repeat-order offers |
| Fading Customers | 182 | 109.0 | 2.20 | 685.99 | Win-back coupon before they are gone for good |

Cluster numbers are only labels with no ordering; names were assigned by rule from the profile and Premium is the segment with the highest mean monetary value.

PCA on the scaled RFM matrix: `explained_variance_ratio_` = 0.634 and 0.218, total 0.852. Output: `outputs/customer_segments.csv` (customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium).

### Lapse problem

Cutoff 2025-09-01. Cohort: every customer with at least one completed order before the cutoff = **1,156 customers, 443 lapsed (38.32%)**. Label `lapsed = 1` if no completed order dated 2025-09-01 to 2025-09-30. Features (orders before the cutoff only plus the clean customer table): recency, frequency, monetary, average discount_pct, cancellation rate, average delivery_minutes, favourite category, city, age, gender, tenure in days. Split with `train_test_split(test_size=0.2, stratify=y, random_state=42)`: **924 train / 232 test**.

Preprocessing: `StandardScaler` on the numeric columns and `OneHotEncoder(drop="first")` (k-1 columns) on city, gender and favourite category, inside the Pipeline. The scaler's learned mean of frequency is 8.115 against a full-cohort mean of 8.029; they differ because the scaler only saw the 924 training rows, not the 232 test rows. A code assertion confirms no feature column has a missing value.

### Models

| Item | Result |
|---|---|
| Baseline: always "not lapsed" (test) | accuracy 61.6%, lapsed-class F1 0.000 |
| LogisticRegression, best (`C` in 0.01..100, cv=5, F1) | C = 100, cv-F1 0.651 |
| RandomForest, best (depth 3/5/8 x leaf 5/10/20, cv=5, F1) | max_depth = 8, min_samples_leaf = 10, cv-F1 0.681 |

Test set (lapsed class as positive):

| Model | Precision | Recall | F1 | Confusion matrix [[TN, FP], [FN, TP]] |
|---|---|---|---|---|
| Logistic regression | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| Random forest | 0.659 | 0.652 | 0.655 | [[113, 30], [31, 58]] |

**Selected model: random forest** (chosen on the best cross-validated F1, not on the test set; test F1 0.655, above the 0.55 requirement). Missing a lapsing customer means losing all of their future revenue, while wrongly targeting a loyal customer costs only a small coupon or message, so recall matters more than precision here. Both models catch the same 58 of 89 lapsing customers, but the forest wastes two fewer offers and has the better cross-validated F1, so it is the safer default.

### Class-balance experiment

The training set was over-sampled by adding randomly drawn duplicates of lapsed rows (`random_state=42`) until both classes had 570 rows; the best logistic regression was refitted on it and scored on the unchanged test set.

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| Original training data | 0.644 | 0.652 | 0.648 |
| Over-sampled training data | 0.589 | 0.854 | 0.697 |

Recall moved most (+0.20) while precision did not rise (it fell slightly). For the retention team this means over-sampling finds many more true leavers, at the price of contacting more customers who would have stayed; it is the right choice if offers are cheap and missed leavers are expensive.

### Interpretation

Top 5 random-forest feature importances: frequency 0.280, monetary 0.271, recency 0.115, average delivery time 0.061, cancellation rate 0.052.

Logistic regression coefficients (standardised features):
- frequency -1.697: customers who order often are much less likely to lapse, so frequency is the main loyalty signal.
- monetary +0.283: given the same frequency, higher spend is slightly linked to lapsing; spend and frequency are correlated, so do not over-read this one.
- cancel_rate -0.124: small and not meaningful in this synthetic data; cancellations do not push customers away here.

The selected model was saved with `joblib.dump` to `outputs/lapse_model.joblib`; reloading it with `joblib.load` reproduces the original test predictions exactly (checked in code).

### Retention list

Among the 40 Premium customers in the test set, the 10 with the highest predicted lapse probability are in `outputs/retention_list.csv` (customer_id, lapse_probability, actually_lapsed); **1 of the 10 truly lapsed**. This list is illustrative and not a leakage-free evaluation, because the Premium flag comes from the whole-window RFM, which includes the September window that defines the lapse label. (Premium customers also rarely lapse, so their predicted probabilities are all below 0.5.)

## Part C

Script: `part_c_triage.py` (`python part_c_triage.py` runs every step below and writes both output files). Tickets are read with `keep_default_na=False` and processed in file order. All model calls go through the supplied offline `mock_llm.py`; `mock_llm.reset()` is called immediately before each full run.

### Guardrails and redaction

| Check | Result |
|---|---|
| Empty tickets (`skipped_empty`, never sent to the model) | 4 |
| Long tickets (`flagged_long`, est_tokens > `MAX_INPUT_TOKENS` = 400, never sent) | 4 (each estimates 734 tokens) |
| Emails replaced by `[EMAIL]` | 55 |
| Phone numbers replaced by `[PHONE]` | 91 |
| Order ids replaced by `[ORDER_ID]` | 132 |
| Total substitutions | 278 |
| PII found after re-applying the three patterns to the redacted texts | 0 |

The patterns are applied in the order email, phone, order id. The user message sent to the model is exactly the redacted ticket text. The script asserts, for every prompt actually sent, that no pattern matches it.

### Prompts

Both prompts contain `CLASSIFY`, list the four categories and demand JSON only (`{"category": "...", "reason": "..."}`). The zero-shot prompt (about 29 estimated tokens) has no `EXAMPLES:`; the few-shot prompt (about 99 estimated tokens) has an `EXAMPLES:` line followed by four labelled examples, one of them fraud.

### Retries and validation

`call_with_retry` allows at most 3 retries per ticket with a wait of `BASE_DELAY x 2**n x random.uniform(0.9, 1.15)`, `BASE_DELAY` = 0.05 s, and prints one line per retry (ticket id, retry number, wait). A full run over the 156 eligible tickets hits **19** `TransientLLMError`s, all recovered on the first retry (every retry is retry number 0; waits between 0.0453 and 0.0562 s, inside the 0.045 to 0.0575 s range). No ticket ended with a permanent error.

Reply validation: valid JSON with an allowed category is `classified_json`; otherwise the lowercase words from `re.findall(r"[a-z]+", ...)` are checked and exactly one allowed category word gives `classified_keyword`; anything else, including valid JSON with a category outside the list, is `flagged_invalid_output` and goes to human review. Nothing is guessed.

| Status (one full run) | Tickets |
|---|---|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

### Zero-shot versus few-shot (no token budget)

| Run | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126 of 147) | 8,932 | 60.8 |
| Few-shot | 147 | 100.0% (147 of 147) | 19,852 | 135.0 |

Zero-shot confusion table (rows = true, columns = predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

Every zero-shot error is a fraud ticket sent to billing (11) or a technical ticket sent to billing (10). The few-shot prompt costs about 74 more tokens per ticket (135.0 against 60.8, roughly 2.2 times) and 10,920 more tokens in total, and in return 21 more tickets are right, which is about 520 extra tokens per corrected ticket. That is worth paying when a wrong label is expensive: 11 of the 34 fraud tickets were misrouted to billing under zero-shot, and fraud has a 2-hour security deadline. It is not worth paying for low-risk bulk routing where a human re-checks the queue anyway. The mock classifier is keyword-based, so the 100% few-shot figure shows the plumbing works, not that a real model would be perfect.

### Token budget demo

Zero-shot with `TOKEN_BUDGET = 6000` (checked before every call): **51 tickets** end `budget_exhausted` and go to human review; 6,023 tokens were spent (the call that crosses the budget finishes, so the total can slightly overshoot). Saved as `outputs/triage_budget_demo.csv`.

### Routing and final output

`outputs/triage_results.csv` (164 rows; columns ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used) comes from the few-shot run with `TOKEN_BUDGET = 100000`, joined to `outputs/customer_segments.csv` from Part B. Rule order: skipped_empty / flagged_long / flagged_invalid_output / budget_exhausted go to `human_review` (NORMAL); otherwise fraud goes to `security_team` (HIGH); otherwise a Premium customer goes to `premium_desk` (HIGH); otherwise `support_queue` (NORMAL). A customer with no segment row counts as not premium.

| Route | Priority | Tickets |
|---|---|---|
| support_queue | NORMAL | 102 |
| security_team | HIGH | 34 |
| human_review | NORMAL | 17 |
| premium_desk | HIGH | 11 |

Rule-validated versus human: the 147 tickets with a parseable, allowed category are routed automatically; the other 17 (9 invalid model replies, 4 empty, 4 too long) go to humans. "Rule-validated" means the reply passed the format and category check, not that its label was independently verified.

## Part D

Scripts: `part_d_retrieval.py` (chunking, TF-IDF retrieval, answers, evaluation) and `part_d_agent.py` (tools, ReAct loop, safe stopping, checkpoints, scenario runner). Every refund here is **simulated**: no payment service exists and the "ledger" is a local file.

### Chunking and retrieval

The five policy files are split into 40-word chunks with 10 words of overlap (first chunk words 0 to 39, next starts 30 words later, stop at the first chunk that reaches the last word), with the source file name kept on every chunk. Result: **15 chunks** (refund 4, delivery 3, escalation 3, payment 3, privacy 2). `TfidfVectorizer()` is fitted on the chunk texts, queries are transformed with the same vectorizer and ranked by `cosine_similarity`; `retrieve(query, k=3)` returns dictionaries with source, score and text. For "refund policy late delivery" all three top chunks come from `refund_policy.txt`.

TF (term frequency) measures how often a word appears in a chunk, and IDF (inverse document frequency) lowers the weight of words that appear in many chunks, so rare, informative words count more. A limitation is that TF-IDF only matches the exact words used, so a question and a policy sentence that mean the same thing in different words ("money back" against "refund") can score low.

`answer(question)` retrieves three chunks and sends two messages to the mock: a system message containing `POLICY ASSISTANT` (answer only from the context, otherwise say it could not find the information) and a user message `CONTEXT:`, the chunks each preceded by `[<file name>]` on its own line, then `QUESTION:` and the question. It returns the text and the sources. The assignment fixes "always retrieve three chunks", so there is no extra similarity threshold; questions with no evidence get the mock's fallback answer.

### Evaluation (14 questions in `policy_eval.csv`)

| Chunking | Expected doc in top-3 (of 12) | Answer contains expected fact (of 12) | Fallback on unanswerable (of 2) |
|---|---|---|---|
| overlap 10 (15 chunks) | 12 | 12 | 2 |
| overlap 0 (13 chunks) | 12 | **10** | 2 |

With overlap 0, two questions break because the answer sentence is cut at a chunk boundary: "What delivery time makes a completed order eligible for a full refund?" returns only "receives a full refund of the order amount." (the condition "more than 45 minutes" ended up in the previous chunk), and "How soon is money returned when a payment fails but is deducted?" returns the sentence without "within 48 hours" (that part started the next chunk). Overlap keeps the whole sentence together in at least one chunk.

### Tools

`TOOLS_SPEC` lists five tools (name, one-sentence description, JSON-style parameters) and `TOOL_MAP` maps each name to its Python function.

| Tool | What it does |
|---|---|
| `get_order(order_id)` | Reads `data/instamart.db` with a parameterised query; returns order_id, status, order_amount, delivery_minutes, or `{"error": "order not found"}` |
| `search_policy(query)` | Returns the `retrieve` output under `results` |
| `calculate_refund(order_amount, delivery_minutes)` | More than 45 minutes: 100% of the amount; otherwise 0 |
| `issue_refund(order_id, refund_inr)` | Simulated refund up to INR 1000; appends one JSON line to `outputs/refund_ledger.jsonl` |
| `escalate_to_human(order_id, refund_inr)` | Sends a refund above INR 1000 to a human lead; appends one ledger line |

Safety checks added on top of the brief: model-generated tool names and arguments are validated against `TOOLS_SPEC` (unknown tool, missing or extra keys and wrong types are rejected); the customer id for ownership comes from the verified session, never from the model, and another customer's order looks exactly like "not found"; `issue_refund` and `escalate_to_human` re-read the order from the database and refuse any refund that does not match the policy (wrong amount, on-time, cancelled, or the wrong tool for the amount); and the ledger is idempotent per (tool, order id), so asking twice never writes a second refund (the second call returns `duplicate: true`). Tool exceptions become `{"error": ...}` observations and are never reported as success.

### ReAct loop and the seven scenarios

`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` starts from a system message containing `AGENT` plus the user message, calls the model, executes the requested tool through `TOOL_MAP`, appends the assistant and tool messages and repeats until the model gives a final answer; the status is the word before the colon. One line per run goes to `outputs/agent_runs.jsonl`.

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,177 |
| S2 | REFUND | 135.00 | 5 | 1,177 |
| S3 | ESCALATE | 1999.00 | 5 | 1,164 |
| S4 | DENIED | 0.00 | 4 | 829 |
| S5 | DENIED | 0.00 | 3 | 505 |
| S6 | NOT_FOUND | 0.00 | 2 | 184 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,336 |

All refund amounts equal `expected_refund_inr` within 0.01; S7 uses all six model calls and stays under the default 6000-token limit.

### Stopping safely

| Demonstration | Status | Detail |
|---|---|---|
| S1 with `token_limit=200` | FALLBACK_TOKEN_LIMIT | stopped after 2 model calls (214 tokens) before running the second tool; only `get_order` had executed |
| S1 with `time_limit_seconds=0` | FALLBACK_TIMEOUT | stopped before the first model call (0 calls) |
| S7 (loop) | FALLBACK_MAX_ITERATIONS | stopped after 6 model calls |

Every fallback returns the steps completed so far and a short `HANDOFF to human agent: ...` summary in `final_answer`.

### Checkpoint and resume

After every tool observation a checkpoint is written (atomically) to `state/<run_id>.json` with the run id, scenario id, full message list, tokens used, step number and the tools executed so far. `resume_agent(run_id)` reloads it and continues. In the S1 demonstration a simulated crash was raised right after the second tool's checkpoint; `resume_agent` then finished the run with status REFUND and 128.25 in 5 model calls in total.

Tools executed across the crash and the resume, in order: **`get_order`, `search_policy`, `calculate_refund`, `issue_refund`** (four executions, none repeated). In this particular demo the main S1 run had already written the S1 refund to the ledger, so the resumed `issue_refund` returned `duplicate: true` and wrote no new line; the final ledger holds 3 lines (S1, S2, S3). The case with an empty ledger, and the worst case where the crash happens after the ledger write but before the checkpoint, are covered by `tests/test_part_d.py`.

### ReAct versus plan-and-execute (about 230 words)

My ReAct agent needed 5 model calls for S1 (1,177 tokens), 3 for S5 and 2 for S6, because it stops as soon as an observation settles the case. Each call resends the whole history, so tokens grow with every step. A plan-and-execute design asks the model once for the full step list and would need roughly 1 planning call plus, at most, one replanning call; I did not build it, so those counts are estimates. It would be cheaper on the happy path (about 1 to 2 calls instead of 5). Its weakness is unexpected observations: a plan written before seeing the order cannot know the order is missing (S6) or cancelled (S5), so it either runs useless steps or must replan on error, adding a call and complexity. Whether to issue, escalate or deny depends on the amount and delivery time that only `get_order` reveals, so the plan would need branches. For this refund workflow I would choose ReAct: the flow is short (2 to 5 calls), the decision depends on live data, early exits are common, and my code-level checks (argument validation, ownership, policy re-check, idempotent ledger) keep each step safe. I would revisit plan-and-execute if workflows grew to many fixed steps where token cost dominates.

## Limitations and Next Steps

- The mock LLM is a keyword matcher, not a real model: the 100% few-shot accuracy and the clean agent behaviour show that the pipeline plumbing and guardrails work, not how a real model would behave. A real model would also need output-schema enforcement and prompt-injection handling.
- Refunds are simulated (a local JSONL ledger). The ledger is idempotent per (tool, order id), so a legitimate second refund for the same order would be blocked by design and would need a new operation id in a real system.
- Customer ownership is checked against the customer id attached to the request (from the scenario file); there is no real login or authentication.
- A resumed agent run restarts its time budget, and checkpoints are local JSON files, not a database.
- TF-IDF only matches exact words (no synonyms); the policy evaluation set is small (14 questions), so 12/12 is encouraging, not proof.
- The project uses a flat layout (scripts in the repository root) rather than a `src/` package, and was run and tested on Linux with Python 3.13; Windows PowerShell commands are given but were not executed on Windows.
- The Premium flag is a post-hoc whole-window label (it uses orders from September, the month that defines the lapse label), so the retention list is illustrative and not a leakage-free evaluation.
- The lapse model has about 232 test customers, so test F1 values move by several points with a different split; the cross-validated scores are the steadier guide.
- The data is synthetic, so the patterns (segments, lapse behaviour, ticket text) are far cleaner than real exports would be.
