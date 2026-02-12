# app.py

import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .search_engine import VectorSearchEngine

engine = VectorSearchEngine(
    pages_dir='pages',
    index_file='index.txt',
    tfidf_lemmas_dir='tfidf_lemmas',
    tfidf_tokens_dir='tfidf_tokens',
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("Загрузка предвычисленных TF-IDF...")
    print("=" * 60)
    engine.build()
    print("\nСервер готов!\n")
    yield
    print("Завершение.")


app = FastAPI(title="Векторный поиск", lifespan=lifespan)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    stats = engine.get_stats()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": stats,
    })


@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str = Query(default="", min_length=0),
    top_k: int = Query(default=20, ge=1, le=100),
):
    results = []
    elapsed = 0.0
    query = q.strip()

    if query:
        start = time.time()
        results = engine.search(query, top_k=top_k)
        elapsed = round(time.time() - start, 4)

    is_htmx = request.headers.get("HX-Request") == "true"

    if is_htmx:
        return templates.TemplateResponse("results.html", {
            "request": request,
            "query": query,
            "results": results,
            "elapsed": elapsed,
            "top_k": top_k,
        })

    stats = engine.get_stats()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": stats,
        "query": query,
        "results": results,
        "elapsed": elapsed,
        "top_k": top_k,
    })


@app.get("/api/search")
async def api_search(
    q: str = Query(default="", min_length=1),
    top_k: int = Query(default=20, ge=1, le=100),
):
    start = time.time()
    results = engine.search(q.strip(), top_k=top_k)
    elapsed = round(time.time() - start, 4)
    return {
        "query": q.strip(),
        "results": results,
        "total": len(results),
        "elapsed": elapsed,
    }


@app.get("/api/stats")
async def api_stats():
    return engine.get_stats()