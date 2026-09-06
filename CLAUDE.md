# Predictive Deployment Risk Assessment System

Scores GitHub Pull Requests for deployment risk (0-1 score + Low/Medium/High
label) and explains the score in plain language with an LLM. Academic project,
2 developers, 8-10 weeks. Single-user, no authentication.

## Stack — do not add anything outside this list without asking first
Backend:  FastAPI, SQLAlchemy, PostgreSQL, pydantic-settings, requests,
          xgboost, scikit-learn, pandas, openai, pytest
Frontend: React (Vite), Tailwind CSS

## Hard rules
1. features.py lives at the backend root and is imported by BOTH
   services/pipeline.py and ml/train.py. There must never be a second
   feature-computation function anywhere in the repo.
2. The XGBoost model loads ONCE at module import in services/model.py.
   Never load it inside a request handler.
3. Feature ordering comes from a FEATURE_ORDER list saved with the model.
   Never rely on dict insertion order.
4. Routes are thin: parse the request, call one service function, return.
   No business logic inside a route handler.
5. No secrets in code. Everything goes through config.Settings reading env vars.
6. user_id comes from config, not from a token.
7. Any service function that touches the network raises a custom exception.
   Never let a raw requests or openai exception escape a service.

## Style
Plain functions over classes. Type hints on public functions.
No comments that restate what the code already says.
