# Build Playbook — Stepwise Prompts
## Predictive Deployment Risk Assessment System

30 prompts, ordered, each with a verification gate. Follow them in sequence and the
engineering will work. The one genuinely uncertain step (model quality) has a
contingency branch at Step 13b.

---

## How to use this

1. **Run Step 0 first.** It writes `CLAUDE.md`, which loads your constraints into every
   later session automatically. Skipping it is the single biggest cause of drift.
2. **One step per prompt.** Do not paste two steps at once. The whole point of the gates
   is that a broken Step 7 never reaches Step 12.
3. **Never advance until the "Done when" passes.** If it doesn't, reply with the actual
   error output rather than starting a new step.
4. **Start a fresh session at each phase boundary** (W1 → W2 → W3 …). Long sessions drift.
5. **If it proposes a library not in `CLAUDE.md`, say no.** Every extra dependency is
   surface area you will have to explain in the viva.
6. **Follow each step with:** *"Verify the Done-when yourself and show me the output."*
   Do not accept "this should work."

---

## Prerequisites — do these before Step 1

| Need | Where | Note |
|---|---|---|
| Python 3.11+ | `python3 --version` | |
| Node 18+ | `node --version` | |
| GitHub Personal Access Token | github.com → Settings → Developer settings → PAT (classic) | `public_repo` scope is enough. Raises your rate limit from 60/hr to 5,000/hr — you cannot do this project without it. |
| OpenAI API key | platform.openai.com | Load $5. Total project cost is under $2. |
| Postgres database | neon.tech free tier | **Use Neon from day one, not local Postgres.** Same database in dev and prod, no install, and `gen_random_uuid()` works out of the box. |

---

# PHASE 1 — Foundation (Week 1)

## Step 0 — Write the constitution

> Create a `CLAUDE.md` at the repo root with exactly this content, and nothing else:
>
> ```
> # Predictive Deployment Risk Assessment System
>
> Scores GitHub Pull Requests for deployment risk (0-1 score + Low/Medium/High label)
> and explains the score in plain language with an LLM. Academic project, 2 developers,
> 8-10 weeks. Single-user, no authentication.
>
> ## Stack — do not add anything outside this list without asking first
> Backend:  FastAPI, SQLAlchemy, PostgreSQL, pydantic-settings, requests,
>           xgboost, scikit-learn, pandas, openai, pytest
> Frontend: React (Vite), Tailwind CSS
>
> ## Hard rules
> 1. `features.py` lives at the backend root and is imported by BOTH
>    `services/pipeline.py` and `ml/train.py`. There must never be a second
>    feature-computation function anywhere in the repo.
> 2. The XGBoost model loads ONCE at module import in `services/model.py`.
>    Never load it inside a request handler.
> 3. Feature ordering comes from a `FEATURE_ORDER` list saved with the model.
>    Never rely on dict insertion order.
> 4. Routes are thin: parse the request, call one service function, return.
>    No business logic inside a route handler.
> 5. No secrets in code. Everything goes through `config.Settings` reading env vars.
> 6. `user_id` comes from config, not from a token.
> 7. Any service function that touches the network raises a custom exception.
>    Never let a raw `requests` or `openai` exception escape a service.
>
> ## Style
> Plain functions over classes. Type hints on public functions.
> No comments that restate what the code already says.
> ```

**Done when:** `CLAUDE.md` exists at the repo root and you have read it start to finish.

---

## Step 1 — Backend skeleton

> Create the backend skeleton. Only these files, nothing more:
>
> - `backend/requirements.txt` — fastapi, uvicorn[standard], sqlalchemy, psycopg2-binary, pydantic-settings, requests, python-dotenv, xgboost, scikit-learn, pandas, openai, pytest, httpx
> - `backend/config.py` — a `Settings` class using pydantic-settings that reads `DATABASE_URL`, `GITHUB_TOKEN`, `OPENAI_API_KEY`, `MODEL_PATH`, `DEFAULT_USER_EMAIL`, `ENV`. Export a module-level `settings = Settings()`.
> - `backend/.env.example` — every key above with placeholder values.
> - `backend/database.py` — SQLAlchemy engine from `settings.DATABASE_URL`, `SessionLocal`, a declarative `Base`, and a `get_db()` generator dependency.
> - `backend/main.py` — FastAPI app, CORS allowing `http://localhost:5173`, and `GET /health` returning `{"status": "ok"}`.
>
> Add a `.gitignore` covering `.env`, `__pycache__/`, `venv/`, `node_modules/`.

**Done when:** `uvicorn main:app --reload` starts clean and `curl localhost:8000/health` returns `{"status":"ok"}`.

---

## Step 2 — Database models

> Create `backend/models.py` with four SQLAlchemy models matching this schema exactly:
>
> *(paste the table from LLD §2)*
>
> Requirements:
> - UUID primary keys with `server_default=text("gen_random_uuid()")`
> - `TIMESTAMPTZ` for every timestamp, `server_default=func.now()` where the schema says so
> - `JSONB` for `predictions.features`
> - `CheckConstraint("risk_score >= 0 AND risk_score <= 1")` on predictions
> - `UniqueConstraint(user_id, github_repo_id)` and `UniqueConstraint(repository_id, github_pr_number)`
> - Index every foreign key, plus `predictions.created_at`
> - Relationships in both directions with `cascade="all, delete-orphan"` on the parent side
>
> Then in `main.py`, add a startup handler that calls `Base.metadata.create_all(bind=engine)`
> and inserts one user row with `settings.DEFAULT_USER_EMAIL` if the users table is empty.
> Add `get_default_user(db)` to `database.py`.

**Done when:** the app starts against your Neon database and all four tables exist with the right columns and constraints. Check in the Neon SQL editor.

---

## Step 3 — Pydantic schemas

> Create `backend/schemas.py` with all request and response models:
>
> - `RepositoryCreate` — `owner: str`, `name: str`
> - `RepositoryOut` — id, owner, name, github_repo_id, connected_at
> - `PullRequestOut` — id, github_pr_number, title, author, state, opened_at, merged_at
> - `PredictionCreate` — `pull_request_id: UUID`
> - `PredictionOut` — id, pull_request_id, risk_score, risk_label, explanation, features (dict), model_version, created_at
>
> All `Out` models use `model_config = ConfigDict(from_attributes=True)`.
> Add validation to `RepositoryCreate`: owner and name must match `^[A-Za-z0-9._-]+$`
> and be 1–100 characters.

**Done when:** `python -c "import schemas"` runs clean and `/docs` shows the models once they are wired to routes.

---

# PHASE 2 — GitHub Integration (Week 2)

## Step 4 — GitHub client

> Create `backend/exceptions.py` with `GitHubAPIError(Exception)` and `PredictionError(Exception)`,
> each carrying a `message` and an optional `status`.
>
> Create `backend/services/github.py` using `requests` with a module-level `Session` carrying:
> ```
> Authorization: Bearer {settings.GITHUB_TOKEN}
> Accept: application/vnd.github+json
> X-GitHub-Api-Version: 2022-11-28
> ```
>
> Public functions:
> - `get_repo(owner, name) -> dict`
> - `list_pull_requests(owner, name, state="all", limit=30) -> list[dict]`
> - `get_pr(owner, name, number) -> dict`
> - `get_pr_files(owner, name, number) -> list[dict]`
> - `get_pr_commits(owner, name, number) -> list[dict]`
>
> One shared `_request(method, path, **kw)` helper that:
> - raises `GitHubAPIError` on any non-2xx, with the GitHub error message included
> - on **403 with `X-RateLimit-Remaining: 0`**, raises immediately with a message naming the reset time from `X-RateLimit-Reset` — never retries into the same wall
> - retries **exactly once** after 1 second on a 5xx or `requests.ConnectionError`
> - has a 15-second timeout on every call
>
> A `_paginate(path, params)` helper that follows the `Link` header with `per_page=100`,
> used by `get_pr_files` and `get_pr_commits`.

**Done when:** a throwaway script fetches a real PR from a small public repo and prints its file count and commit count, and both match what you see on github.com.

---

## Step 5 — Repository routes

> Add to `main.py`:
>
> - `POST /repositories` — body `RepositoryCreate`. Calls `github.get_repo()` to validate the repo exists and is public. If GitHub 404s, return **404** with `{"detail": "Repository not found or not public"}`. On success, upsert a `repositories` row for the default user (unique on `user_id + github_repo_id`) and return `RepositoryOut`.
> - `GET /repositories` — list the default user's repositories, newest first.
>
> Also add two FastAPI exception handlers in `main.py`: `GitHubAPIError` → **502** and
> `PredictionError` → **500**, both returning `{"detail": "<message>"}`. No stack traces
> in responses.

**Done when:** `POST /repositories {"owner":"pallets","name":"flask"}` returns 200 with a row, calling it twice does not create a duplicate, and a nonsense repo returns a clean 404.

---

## Step 6 — Pull request listing

> Add `GET /repositories/{repo_id}/pull-requests` to `main.py`.
>
> It loads the repository row (404 if unknown), calls
> `github.list_pull_requests(owner, name, state="all", limit=30)`, upserts each PR into
> `pull_requests` keyed on `(repository_id, github_pr_number)` — updating title, author,
> state, opened_at, merged_at on conflict — then returns the stored rows as
> `list[PullRequestOut]`, most recently opened first.
>
> Put the upsert logic in a new `services/repo_sync.py` function, not in the route.
>
> Derive `state`: `"merged"` if `merged_at` is set, otherwise the GitHub `state` value.

**Done when:** the endpoint returns 30 PRs for a connected repo, calling it twice leaves 30 rows (not 60), and a merged PR shows `state: "merged"`.

---

# PHASE 3 — Feature Engineering (Week 3)

## Step 7 — The shared feature function

> Create `backend/features.py`. This is the most important file in the project — it is
> imported by both the live API and the offline training script, so it must be pure.
>
> **No network. No database. No `datetime.now()` inside the function** — take a `now`
> parameter with a default of `None` meaning "use `datetime.now(timezone.utc)`", and pass
> it explicitly from training.
>
> ```python
> FEATURE_ORDER = [...]  # the exact list below, in this order
>
> def build_feature_vector(pr: dict, files: list[dict], commits: list[dict],
>                          author_pr_count: int = 0, now=None) -> dict[str, float]:
> ```
>
> Compute exactly these 16 features, all returned as floats:
>
> *(paste the table from LLD §5)*
>
> Rules:
> - Booleans return `0.0` or `1.0`, never `True`/`False`
> - Guard every division against zero
> - Missing or null GitHub fields fall back to `0.0`, never raise
> - Path matching is case-insensitive
> - `is_off_hours` and `is_weekend` use `merged_at` if present, else `now`
>
> Also add `to_array(features: dict) -> list[float]` that orders values by `FEATURE_ORDER`
> and raises `KeyError` naming any missing key.
>
> Then create `backend/tests/test_features.py` with at least 8 tests: an empty PR, a
> tests-included PR, a config-touching PR, a weekend merge, a still-open PR, a
> divide-by-zero case, a missing-field case, and a `to_array` ordering test.

**Done when:** `pytest tests/test_features.py` is fully green and `to_array` raises loudly on a missing key.

---

## Step 8 — Eyeball real features

> Add a temporary debug route `GET /debug/features/{repo_id}/{pr_number}` that fetches the
> PR from GitHub, builds the feature vector, and returns both the raw counts and the
> feature dict side by side.
>
> Mark it clearly with a `# TODO: remove before deployment` comment.

**Done when:** you have manually opened 3 real PRs on github.com and confirmed that `files_changed`, `lines_added`, `lines_removed`, and `commit_count` match exactly what the feature vector reports. **Do not skip this.** Every downstream number depends on it.

---

## Step 9 — Start collecting data NOW (run in the background all of Week 3)

> This step is scheduling, not effort. Read the warning before writing code.

**Why now:** collecting 4,000 PRs needs roughly 3 API calls each — about 12,000 GitHub calls against a 5,000/hour limit. That is **three hours minimum of wall-clock time**, and in practice a full day once you hit secondary rate limits and have to re-run. If you start this in Week 4 as originally planned, you will be blocked. Start it the moment Step 7 passes and let it run while you finish Week 3.

> Create `backend/ml/collect.py` — a standalone script, not part of the FastAPI app.
>
> - Takes a hardcoded list of 12–20 mid-sized public repos (target: 2k–20k stars, active, not monorepos — e.g. `pallets/flask`, `psf/requests`, `encode/httpx`, `tiangolo/fastapi`, `pydantic/pydantic`)
> - For each repo, fetches merged PRs from the last ~2 years, then their files and commits
> - Writes one JSON object per line to `backend/ml/data/raw_prs.jsonl` containing repo, pr, files, commits
> - **Caches aggressively:** before fetching a PR, skip it if its id is already in the output file. The script must be safely re-runnable after a crash.
> - **Handles rate limits by sleeping,** not crashing: on 403 with remaining 0, sleep until `X-RateLimit-Reset` plus 5 seconds, then continue
> - Prints progress every 25 PRs: repo, count so far, and remaining rate limit
> - Also fetches, per repo, the last 500 commits on the default branch into `raw_commits.jsonl` — Step 10 needs these for revert and hotfix detection
>
> Add `backend/ml/data/` to `.gitignore`.

**Done when:** the script has been running for an hour without crashing, `wc -l raw_prs.jsonl` is climbing, and killing it and restarting resumes rather than starting over.

**Target:** 3,000–5,000 merged PRs. Let it finish before Step 10.

---

# PHASE 4 — Training Labels (Week 4)

## Step 10 — Label derivation

> Create `backend/ml/label.py`. It reads `raw_prs.jsonl` and `raw_commits.jsonl` and writes
> `labeled_prs.jsonl` with an added `is_risky` (0/1) and a `label_reason` string.
>
> A PR is labelled risky (1) if **any** signal fires:
>
> **Signal A — reverted.** A later commit on the default branch whose message matches
> `^Revert "` and contains the PR title, or which contains the PR's `merge_commit_sha`.
>
> **Signal B — same-area hotfix.** A commit dated within 7 days after the merge whose
> message matches `\b(fix|hotfix|patch|bug|regression)\b` case-insensitively, **and** which
> touches at least one file path the PR also touched. Both conditions required.
>
> **Signal C — post-merge bug issue.** An issue labelled `bug`, **created after** the PR
> merged, whose body or title references the PR number.
>
> **Critical, do not get this backwards:** a PR that *closes* a bug issue is a fix, not a
> risk. Signal C only counts issues created *after* the merge that *reference* the PR.
> Never count issues the PR closes. If that distinction cannot be made reliably from the
> data you collected, **drop Signal C entirely** and use A and B only — they are unambiguous.
>
> Record which signal fired in `label_reason` so the distribution can be audited.
>
> Print at the end: total PRs, positive count, positive rate, and a breakdown by signal.

**Done when:** the script runs and prints a positive rate. **Expect 5–15%.** If it is under 2% or over 40%, the heuristic is broken — stop and inspect 20 labelled examples by hand before continuing.

---

## Step 11 — Audit the dataset before training

> Do not skip this. It is 30 minutes that can save Week 5.
>
> Create `backend/ml/audit.py` that loads `labeled_prs.jsonl`, builds feature vectors using
> `features.build_feature_vector` (the same function the API uses — import it, do not
> reimplement), and prints:
>
> 1. Row count, positive count, positive rate
> 2. Per-repository positive rate — if one repo contributes most positives, the model will learn that repo, not risk
> 3. For each of the 16 features: mean for positives vs mean for negatives, and the ratio
> 4. Any feature that is constant across the whole dataset (it is dead weight — remove it)
> 5. 5 random positive examples and 5 random negatives, printed with title and `label_reason`
>
> Save the feature vectors to `backend/ml/data/dataset.parquet` so training does not have to
> recompute them.

**Done when:** you have read the output and can name at least three features whose positive and negative means visibly differ. **If no feature separates the classes at all, the labels carry no signal — go to Step 13b now, before training.**

---

# PHASE 5 — The Model (Week 5)

## Step 12 — Baseline first

> Create `backend/ml/train.py`. Start with the baseline, not XGBoost.
>
> - Load `dataset.parquet`
> - **Split by repository, not randomly** — hold out 3 entire repos as the test set. A random split lets the model memorise repo-specific quirks and inflates every metric.
> - Train `sklearn.linear_model.LogisticRegression(class_weight="balanced", max_iter=1000)` on standardised features
> - Print accuracy, precision, recall, ROC-AUC, PR-AUC, and a confusion matrix on the held-out repos
>
> This is the floor. XGBoost has to beat it to be worth using.

**Done when:** you have a baseline ROC-AUC number written down. Anything from 0.55 to 0.70 is a normal baseline.

---

## Step 13 — XGBoost and evaluation

> Extend `train.py` to train `XGBClassifier` alongside the baseline:
>
> - `scale_pos_weight = n_negative / n_positive` to handle the imbalance
> - Modest hyperparameters — `max_depth=4`, `n_estimators=300`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`. This dataset is small; deep trees will overfit.
> - Early stopping on a validation split carved from the training repos
> - Print the same six metrics for both models side by side
>
> Save to `backend/ml/`:
> - `model.json` via `model.save_model()`
> - `feature_order.json` containing `FEATURE_ORDER` exactly as used at training time
> - `metrics.json` with both models' metrics, the row count, the positive rate, the held-out repo names, and a `model_version` string like `xgb-v1`
>
> Commit all three — the report needs to be reproducible.

**Done when:** `metrics.json` exists and XGBoost's ROC-AUC beats the baseline. **If ROC-AUC ≥ 0.75, go to Step 14.** If not, Step 13b.

---

## Step 13b — CONTINGENCY: ROC-AUC below 0.75

Work through these in order, re-evaluating after each. Stop as soon as you clear 0.75.

| Try | Why it might be the problem |
|---|---|
| **1. Widen the hotfix window** from 7 to 14 days in `label.py`, re-label, retrain | Real hotfixes often land the following sprint, not the following week |
| **2. Drop Signal C** if you kept it | It is the signal most likely to be pointing the wrong way |
| **3. Add repo-relative features** — `files_changed` and `total_churn` as a percentile within that repo, not raw counts | A 14-file PR is huge in a small repo and routine in a large one. Raw counts make the model learn repo size instead of risk. **This is the single highest-value fix.** |
| **4. Collect more repos** — go from 12 to 25 | Under ~200 positive examples, XGBoost has very little to learn from |
| **5. Restrict to Signal A only** (reverts) and accept a lower positive rate | Noisy labels cap achievable AUC. A clean 3% positive rate can beat a noisy 12% |

**If you have tried all five and are still under 0.75, that is a finding, not a failure.**
Report the honest number, show the baseline comparison, and write up *why* proxy labels
cap what is learnable from public data alone. Frame the limitation clearly — a well-analysed
0.68 with a correct methodology defends better in a viva than an unexplained 0.82 that came
from a leaky random split. Tell your mentor early rather than at the demo.

---

## Step 14 — Serving the model

> Create `backend/services/model.py`:
>
> - At **module import** (not inside a function, not inside a request handler), load `ml/model.json` into an `XGBClassifier` and read `ml/feature_order.json` into `FEATURE_ORDER`, and read `model_version` from `metrics.json`
> - **Assert at import** that the loaded `FEATURE_ORDER` equals `features.FEATURE_ORDER`, and raise a clear error naming the mismatch if not. This single assertion prevents the worst silent bug in the project.
> - `predict(feature_dict) -> tuple[float, str]` — orders values with `features.to_array()`, calls `predict_proba`, returns `(float(prob), label)`
> - `score_to_label(score)` — `< 0.34` Low, `< 0.67` Medium, else High. Thresholds as module constants.
>
> Add `backend/tests/test_model.py`: a hand-crafted obviously-risky vector (50 files, no
> tests, weekend, config touched) must score **higher** than an obviously-safe one (2 files,
> tests present, weekday, small churn). Test all three label thresholds at their boundaries.

**Done when:** `pytest tests/test_model.py` is green, and adding a fake feature to `features.FEATURE_ORDER` makes the app fail loudly at startup rather than silently mis-predicting.

---

# PHASE 6 — Explanations & Pipeline (Week 6)

## Step 15 — LLM explanations

> Create `backend/services/explain.py`.
>
> One module-level `PROMPT_TEMPLATE` constant. The prompt must:
> - State the role: an engineer explaining a deployment risk score to a teammate
> - Include the score, the label, and **only the 5 features with the largest deviation from the dataset mean** (hardcode the means from `metrics.json` — sending all 16 produces vague output)
> - Instruct: name the top 2–3 drivers, reference only the numbers given, 2–4 sentences, no bullet points, no invented facts about what the code does
> - Instruct explicitly: do not claim certainty the score does not support; a 0.55 is "moderately elevated," not "dangerous"
>
> `generate_explanation(features, score, label) -> str` calls the OpenAI Chat Completions
> API with `temperature=0.3` and `max_tokens=200`.
>
> **On any OpenAI failure — timeout, rate limit, bad key — catch it, log a warning, and
> return a deterministic template string** built from the top features. A failed LLM call
> must never fail the prediction. Return the fallback, not an exception.

**Done when:** calling it with a hand-made high-risk vector produces a coherent paragraph that names real numbers from the vector, and temporarily breaking the API key still returns a usable fallback string instead of a 500.

---

## Step 16 — Faithfulness check

> Write `backend/ml/check_explanations.py`: a script that runs 10 varied feature vectors
> through `generate_explanation` and prints each vector next to its explanation.

**Done when:** you have read all 10 and confirmed that **no explanation states a number that is not in its vector**, and that low-risk vectors do not get alarming language. If any hallucinate, tighten the prompt and re-run — do not move on. This is one of the four things your PRD promises to demonstrate.

---

## Step 17 — The orchestrator

> Create `backend/services/pipeline.py` with one public function:
>
> ```python
> def run_prediction(db, pull_request_id) -> Prediction
> ```
>
> Steps, in order:
> 1. Load the `pull_requests` row and its repository (raise `PredictionError` if unknown)
> 2. `github.get_pr` + `get_pr_files` + `get_pr_commits`
> 3. Compute `author_pr_count` — the author's prior merged PRs in that repo, from the local `pull_requests` table, not a GitHub call
> 4. `features.build_feature_vector(...)`
> 5. `model.predict(features)`
> 6. `explain.generate_explanation(...)`
> 7. Insert a `predictions` row with features, score, label, explanation, and `model_version`
> 8. Commit and return the row
>
> Log one INFO line per stage with elapsed milliseconds, using the `logging` module.
>
> This is the only place these services are called together. Nothing else may call them
> in sequence.

**Done when:** calling `run_prediction` from a script against a real PR inserts exactly one row and logs 6 timing lines.

---

## Step 18 — Prediction and history endpoints

> Add to `main.py`:
>
> - `POST /predictions` — body `PredictionCreate`, calls `pipeline.run_prediction`, returns `PredictionOut`
> - `GET /predictions/pull-request/{pr_id}` — that PR's predictions, newest first
> - `GET /repositories/{repo_id}/predictions` — all predictions across a repo, newest first, joined through `pull_requests`, limit 50
>
> All three are thin — no logic beyond loading and delegating.
>
> Then remove the `/debug/features` route from Step 8.

**Done when:** the full flow works end to end via `/docs`: connect a repo, list PRs, POST a prediction, and see it in both history endpoints. Note the total response time.

---

# PHASE 7 — Frontend (Week 7)

## Step 19 — Scaffold and API client

> Create the frontend with Vite + React + Tailwind:
>
> ```
> npm create vite@latest frontend -- --template react
> ```
>
> then Tailwind per the current Vite guide. Add `frontend/.env` with `VITE_API_URL=http://localhost:8000`.
>
> Create `src/api.js` — one wrapper around `fetch`:
> - Prefixes `import.meta.env.VITE_API_URL`
> - Sets `Content-Type: application/json` on writes
> - Parses JSON, and on a non-ok response throws `new Error(body.detail || "Request failed")`
> - Exports `getRepos`, `connectRepo`, `getPullRequests`, `predict`, `getPRHistory`, `getRepoHistory`
>
> Create `src/useApi.js` — a `useApi(fn, deps)` hook returning `{data, loading, error, refetch}`.
>
> **No component may call `fetch` directly.**

**Done when:** `npm run dev` serves a page that calls `getRepos()` and renders the raw JSON.

---

## Step 20 — Dashboard page

> Create `src/pages/Dashboard.jsx`:
> - A connect-repo form taking `owner/name` in one input, split on `/`, with client-side validation and an inline error message on failure
> - A list of connected repositories; clicking one selects it
> - When a repo is selected, the PR list for it, each row showing number, title, author, state, and a link to open it
> - Clicking a PR navigates to `/pr/:id`
>
> Use React Router. Handle three states explicitly: loading (skeleton rows), empty
> ("No repositories connected yet — add one above"), and error (the message from the API,
> plus a Retry button).

**Done when:** you can connect a repo and see its PRs in the browser, and a bad repo name shows a readable inline error rather than a blank screen or a console-only failure.

---

## Step 21 — PR detail and risk components

> Create three presentational components — props in, markup out, **no data fetching inside any of them**:
>
> - `src/components/RiskBadge.jsx` — takes `score` and `label`, renders a coloured badge plus the score to two decimals. Green / amber / red by label.
> - `src/components/ExplanationPanel.jsx` — takes `explanation` and `features`, renders the paragraph and a compact table of the feature values.
> - `src/components/HistoryTable.jsx` — takes `predictions`, renders date, score, label, model version.
>
> Then `src/pages/PRDetail.jsx`: PR metadata at the top, an **Analyze** button calling
> `POST /predictions`, and on success `RiskBadge` + `ExplanationPanel` + `HistoryTable`
> for that PR.
>
> **The Analyze button takes several seconds** because of the OpenAI call. Show a spinner
> with staged text ("Fetching PR data…" → "Scoring…" → "Generating explanation…") on a
> timer, disable the button while in flight, and never let a double-click fire two requests.

**Done when:** clicking Analyze on a real PR shows a score, a coloured badge, a readable explanation, and the new row appearing in the history table below.

---

## Step 22 — Polish pass

> Go through the whole frontend and fix, in this order:
> 1. Every list has a real empty state with a sentence saying what to do next
> 2. Every async action has a loading state and a disabled control while in flight
> 3. Every error path shows the API's `detail` message, never a blank screen or a raw stack
> 4. The layout works at 375px wide — no horizontal scroll on the page body
> 5. Buttons and links have visible keyboard focus states
> 6. No `console.log` left anywhere

**Done when:** you have opened the app, disconnected your wifi, and clicked through every screen without seeing a blank page or an unhandled crash.

---

# PHASE 8 — Testing & Deployment (Week 8)

## Step 23 — Test suite

> Create `backend/tests/test_api.py` using FastAPI's `TestClient`:
> - `GET /health` returns 200
> - `POST /repositories` with GitHub mocked returns 200 and creates one row
> - `POST /repositories` twice creates only one row
> - `POST /repositories` with GitHub raising 404 returns 404 with a `detail`
> - `POST /predictions` with GitHub and OpenAI both mocked returns 200 and inserts one prediction
> - Both history endpoints return the inserted prediction
>
> Mock `services.github` and `services.explain` with `unittest.mock.patch` — **no real
> network calls in the test suite.** Use a separate test database or a transactional
> rollback fixture.
>
> Add a `conftest.py` with `client` and `db` fixtures.

**Done when:** `pytest` runs all three test files green in under 10 seconds with your wifi off.

---

## Step 24 — Hardening pass

> Review the whole backend and fix:
> 1. Every route that takes an id returns a clean 404 for an unknown one
> 2. No raw `requests` or `openai` exception can escape a service — check every `except`
> 3. `logging` is configured once in `main.py` at INFO with timestamps
> 4. `/docs` shows every endpoint with correct request and response models
> 5. `.env` is gitignored and `.env.example` lists every key `config.py` reads
> 6. Nothing in the repo contains a real token — grep for `ghp_` and `sk-`
> 7. Every foreign key and `predictions.created_at` is actually indexed
>
> Then time the pipeline: log total milliseconds with and without the OpenAI call.

**Done when:** grep finds no secrets, and you have a measured end-to-end number to quote against the PRD's 800 ms target.

---

## Step 25 — Deploy

> Deploy in this order, verifying each before the next:
>
> **1. Database** — already on Neon. Copy the pooled connection string.
>
> **2. Backend to Railway (or Render):**
> - Add `backend/Procfile`: `web: uvicorn main:app --host 0.0.0.0 --port $PORT`
> - Pin versions in `requirements.txt` (`pip freeze`)
> - Set env vars in the dashboard: `DATABASE_URL`, `GITHUB_TOKEN`, `OPENAI_API_KEY`, `MODEL_PATH`, `DEFAULT_USER_EMAIL`, `ENV=production`
> - **Confirm `ml/model.json` and `ml/feature_order.json` are committed** — they are not gitignored, and the app will not start without them
> - Update CORS to allow the Vercel domain
>
> **3. Frontend to Vercel:**
> - Root directory `frontend`, build `npm run build`, output `dist`
> - Set `VITE_API_URL` to the Railway URL
> - Add a `vercel.json` rewrite so React Router deep links work
>
> After deploying, hit the live `/health`, then run one full prediction against the
> deployed stack.

**Done when:** a full prediction works on the live URLs from a browser that has never seen localhost. **Then open the site on your phone** — that is what the demo will feel like.

---

## Step 26 — Documentation

> Write `README.md` at the repo root:
> - One-paragraph description and a screenshot of the dashboard
> - Live demo link and architecture diagram
> - Local setup: prerequisites, backend steps, frontend steps, env vars
> - How to retrain: collect → label → audit → train, with the commands
> - **Model card:** dataset size, repos used, positive rate, how labels were derived, held-out metrics from `metrics.json`, and an honest "Limitations" section on proxy-label noise
> - Project structure tree and API endpoint table
>
> Then update `LLD.md` to match whatever the code actually became. Documentation that
> contradicts the code is worse than none.

**Done when:** someone who has never seen the project can clone it and run it locally from the README alone. Test this on a classmate.

---

# PHASE 9 — Buffer & Demo (Weeks 9–10)

## Step 27 — Feature importance (highest-value stretch goal)

> Add SHAP explanations:
> - `pip install shap`, compute a `TreeExplainer` once at startup in `services/model.py`
> - Return the top 5 contributing features with signed SHAP values in `PredictionOut` under a new `contributions` key
> - Store them inside the existing `features` JSONB under a `_shap` key — no migration needed
> - Add `src/components/ContributionBars.jsx`: a horizontal diverging bar chart, red bars pushing risk up, green pulling it down, labelled with real feature names
>
> Also feed the top SHAP features into the explanation prompt instead of the
> deviation-from-mean heuristic from Step 15 — the explanation becomes grounded in what
> the model actually used.

**Done when:** the PR detail page shows which features drove the score, and the numbers agree with the explanation text.

---

## Step 28 — Batch scan (second stretch goal)

> Add `POST /repositories/{repo_id}/scan` that runs `run_prediction` for every open PR in
> the repo, sequentially with a small delay, and returns a list of `PredictionOut` sorted
> by score descending. Add a "Scan all open PRs" button to the Dashboard that renders the
> result as a ranked risk list.

**Done when:** one click produces a ranked list of every open PR in a real repository. This is the single most impressive thing you can show in the demo.

---

## Step 29 — Demo rehearsal

Not a prompt. Do this with the app running.

- [ ] Full run-through end to end on the **deployed** URL, timed. Target: under 6 minutes.
- [ ] Pick your demo repo in advance and pre-connect it. Have a known-high-risk PR and a known-low-risk PR chosen.
- [ ] `metrics.json` open in a tab, ready to show
- [ ] Screen-recorded backup video, in case the wifi fails
- [ ] Rehearse the answer to: *"How do you know these labels are correct?"* → proxy signals, named limitation, baseline comparison
- [ ] Rehearse: *"What stops the LLM from making things up?"* → structured prompt with only real values, plus the Step 16 check
- [ ] Rehearse: *"Why XGBoost?"* → tabular data, small dataset, beat the logistic baseline by X
- [ ] Rehearse: *"What would you do with another month?"* → real incident labels, calibration, CI integration as a PR check

---

## Step 30 — Final check against the PRD

Confirm all 8 Must-Haves are demonstrable on the live URL:

- [ ] 1. GitHub repository integration
- [ ] 2. Pull request analysis
- [ ] 3. Commit metadata extraction
- [ ] 4. Feature engineering pipeline
- [ ] 5. ML deployment risk prediction
- [ ] 6. AI-generated explanations
- [ ] 7. Prediction history
- [ ] 8. Dashboard visualization

Plus the evaluation-readiness items: passing test suite, `metrics.json` with held-out
ROC-AUC, live deployed link, documented public repo.

---

## The five failure modes these gates exist to prevent

| Failure | Prevented by |
|---|---|
| Training-serving skew — model behaves differently live than in evaluation | Steps 7 and 14: one shared function, plus a startup assertion on `FEATURE_ORDER` |
| Blocked in Week 4 waiting on GitHub rate limits | Step 9: start collection in Week 3, cached and resumable |
| Discovering in Week 5 that the labels carry no signal | Step 11: audit before training, with Step 13b as the branch |
| Hallucinated explanations found at the demo | Step 16: check 10 explanations against their vectors in Week 6 |
| Inflated metrics from a leaky random split | Step 12: split by repository, never randomly |
