from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
from dashboard import dash_app  # from dashboard.py

app = FastAPI()
app.mount("/", WSGIMiddleware(dash_app.server))