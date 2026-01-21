
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from pyserini.encode import AutoQueryEncoder
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import (
    FastAPI,
    Request,
    Depends,
    HTTPException
)
import ray
import json
import time
import asyncio
import uvicorn



@ray.remote(num_cpus=2, max_concurrency=20)
class FaissSearcherActor:
    def __init__(self, faiss_index_path: str, embedder_model: str):
        self.query_encoder = AutoQueryEncoder(embedder_model, l2_norm=True)
        self.faiss_searcher = FaissSearcher(faiss_index_path, self.query_encoder)
        self.faiss_nprobe = 16
        self.faiss_searcher.index.nprobe = self.faiss_nprobe
    
    def search(self, query: str, k: int):
        t0 = time.time()
        results = self.faiss_searcher.search(query, k)
        t1 = time.time()
        print(f"[FaissSearcherActor] Search time: {t1 - t0}s")
        return results


@ray.remote(num_cpus=2, max_concurrency=20)
class LuceneSearcherActor:
    def __init__(self, lucene_index_path: str):
        self.lucene_searcher = LuceneSearcher(lucene_index_path)
    
    def fetch(self, segid: str):
        return json.loads(self.lucene_searcher.doc(segid).raw())


class AsyncBusyRetriever:
    def __init__(self,
                 faiss_index_path: str,
                 embedder_model: str,
                 lucene_index_path: str,
                 num_faiss_actors: int,
                 num_lucene_actors: int,
                 max_search_concurrent: int,
                 max_fetch_concurrent: int,
                 search_timeout_seconds: int,
                 fetch_timeout_seconds: int):
        self.faiss_actors = [
            FaissSearcherActor.remote(faiss_index_path, embedder_model) for _ in range(num_faiss_actors)
        ]
        self.lucene_actors = [
            LuceneSearcherActor.remote(lucene_index_path) for _ in range(num_lucene_actors)
        ]
        
        self._faiss_load = [0] * num_faiss_actors
        self._faiss_lock = asyncio.Lock()
        self._faiss_search_semaphore = asyncio.Semaphore(max_search_concurrent)
        self._search_timeout_seconds = search_timeout_seconds

        self._lucene_load = [0] * num_faiss_actors
        self._lucene_lock = asyncio.Lock()
        self._lucene_fetch_semaphore = asyncio.Semaphore(max_fetch_concurrent)
        self._fetch_timeout_seconds = fetch_timeout_seconds
    
    async def _pick_faiss_actor(self):
        async with self._faiss_lock:
            idx = min(range(len(self.faiss_actors)), key=lambda i: self._faiss_load[i])
            self._faiss_load[idx] += 1
            return idx, self.faiss_actors[idx]
    
    async def _pick_lucene_actor(self):
        async with self._lucene_lock:
            idx = min(range(len(self.lucene_actors)), key=lambda i: self._lucene_load[i])
            self._lucene_load[idx] += 1
            return idx, self.lucene_actors[idx]
    
    async def async_search(self, query: str, k: int):
        try:
            async with self._faiss_search_semaphore:
                idx, actor = await self._pick_faiss_actor()
                print(f"[AsyncBusyRetriever] Picked faiss actor {idx} with load {self._faiss_load[idx]}")
                try:
                    obj_ref = actor.search.remote(query, k)
                    results = await asyncio.wait_for(obj_ref, timeout=self._search_timeout_seconds)
                    return results
                finally:
                    async with self._faiss_lock:
                        self._faiss_load[idx] -= 1
        except asyncio.TimeoutError:
            print(f"[AsyncRetriever] Error in async_search: timeout after {self._search_timeout_seconds}s")
            return None
        except Exception as e:
            print(f"[AsyncRetriever] Error in async_search: {e}")
            return None

    async def async_fetch(self, segid: str):
        try:
            async with self._lucene_fetch_semaphore:
                idx, actor = await self._pick_lucene_actor()
                print(f"[AsyncBusyRetriever] Picked lucene actor {idx} with load {self._lucene_load[idx]}")
                try:
                    obj_ref = actor.fetch.remote(segid)
                    results = await asyncio.wait_for(obj_ref, timeout=self._fetch_timeout_seconds)
                    return results
                finally:
                    async with self._lucene_lock:
                        self._lucene_load[idx] -= 1
        except asyncio.TimeoutError:
            print(f"[AsyncRetriever] Error in async_fetch: timeout after {self._fetch_timeout_seconds}s")
            return None
        except Exception as e:
            print(f"[AsyncRetriever] Error in async_fetch: {e}")
            return None



class AsyncRRRetriever:
    def __init__(self,
                 faiss_index_path: str,
                 embedder_model: str,
                 lucene_index_path: str,
                 num_faiss_actors: int,
                 num_lucene_actors: int,
                 max_search_concurrent: int,
                 max_fetch_concurrent: int,
                 search_timeout_seconds: int,
                 fetch_timeout_seconds: int):
        self.faiss_actors = [
            FaissSearcherActor.remote(faiss_index_path, embedder_model) for _ in range(num_faiss_actors)
        ]
        self.lucene_actors = [
            LuceneSearcherActor.remote(lucene_index_path) for _ in range(num_lucene_actors)
        ]

        self._faiss_rr_index = 0
        self._faiss_rr_lock = asyncio.Lock()
        self._faiss_search_semaphore = asyncio.Semaphore(max_search_concurrent)
        self._search_timeout_seconds = search_timeout_seconds

        self._lucene_rr_index = 0
        self._lucene_rr_lock = asyncio.Lock()
        self._lucene_fetch_semaphore = asyncio.Semaphore(max_fetch_concurrent)
        self._fetch_timeout_seconds = fetch_timeout_seconds
    
    async def _pick_faiss_actor(self):
        async with self._faiss_rr_lock:
            actor = self.faiss_actors[self._faiss_rr_index]
            self._faiss_rr_index = (self._faiss_rr_index + 1) % len(self.faiss_actors)
            return actor
    
    async def _pick_lucene_actor(self):
        async with self._lucene_rr_lock:
            actor = self.lucene_actors[self._lucene_rr_index]
            self._lucene_rr_index = (self._lucene_rr_index + 1) % len(self.lucene_actors)
            return actor
    
    async def async_search(self, query: str, k: int):
        try:
            async with self._faiss_search_semaphore:
                actor = await self._pick_faiss_actor()
                obj_ref = actor.search.remote(query, k)
                results = await asyncio.wait_for(obj_ref, timeout=self._search_timeout_seconds)
                return results
        except asyncio.TimeoutError:
            print(f"[AsyncRetriever] Error in async_search: timeout after {self._search_timeout_seconds}s")
            return None
        except Exception as e:
            print(f"[AsyncRetriever] Error in async_search: {e}")
            return None

    async def async_fetch(self, segid: str):
        try:
            async with self._lucene_fetch_semaphore:
                actor = await self._pick_lucene_actor()
                obj_ref = actor.fetch.remote(segid)
                results = await asyncio.wait_for(obj_ref, timeout=self._fetch_timeout_seconds)
                return results
        except asyncio.TimeoutError:
            print(f"[AsyncRetriever] Error in async_fetch: timeout after {self._fetch_timeout_seconds}s")
            return None
        except Exception as e:
            print(f"[AsyncRetriever] Error in async_fetch: {e}")
            return None


def create_lifespan(faiss_index_path: str,
                    embedder_model: str,
                    lucene_index_path: str,
                    num_faiss_actors: int,
                    num_lucene_actors: int,
                    max_search_concurrent: int,
                    max_fetch_concurrent: int,
                    search_timeout_seconds: int,
                    fetch_timeout_seconds: int):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async_retriever = AsyncBusyRetriever(
            faiss_index_path,
            embedder_model,
            lucene_index_path,
            num_faiss_actors,
            num_lucene_actors,
            max_search_concurrent,
            max_fetch_concurrent,
            search_timeout_seconds,
            fetch_timeout_seconds
        )
        app.state.async_retriever = async_retriever
        print("--- Retriever Server Lifespan Started ---")
        yield
        print("--- Retriever Server Lifespan Ended ---")
    return lifespan


faiss_index_path = ""
embedder_model = ""
lucene_index_path = ""
num_faiss_actors = 10
num_lucene_actors = 2
max_search_concurrent = 30
max_fetch_concurrent = 30
search_timeout_seconds = 60
fetch_timeout_seconds = 60

app = FastAPI(lifespan=create_lifespan(
    faiss_index_path,
    embedder_model,
    lucene_index_path,
    num_faiss_actors,
    num_lucene_actors,
    max_search_concurrent,
    max_fetch_concurrent,
    search_timeout_seconds,
    fetch_timeout_seconds
))

def get_busy_async_retriever(request: Request) -> AsyncBusyRetriever:
    return request.app.state.async_retriever

AsyncBusyRetrieverDep = Annotated[AsyncBusyRetriever, Depends(get_busy_async_retriever)]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/search")
async def search(request: Request, async_retriever: AsyncBusyRetrieverDep):
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in search: {e}")
    
    query = body.get("query")
    k = body.get("k", 10)
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    if not k:
        raise HTTPException(status_code=400, detail="K is required")
    
    response = {}
    try:
        start_time = time.time()
        hits = await async_retriever.async_search(query, k)
        response["hits"] = [{"docid": hit.docid, "score": hit.score.item()} for hit in hits]
        end_time = time.time()
        print(f"[AsyncRetriever] Search time: {end_time - start_time}s")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in search: {e}")
    
    return response
    


@app.post("/fetch")
async def fetch(request: Request, async_retriever: AsyncBusyRetrieverDep):
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in fetch: {e}")
    
    segid = body.get("segid")
    if not segid:
        raise HTTPException(status_code=400, detail="Segid is required")
    
    response = {}
    try:
        start_time = time.time()
        segment = await async_retriever.async_fetch(segid)
        response["segment"] = segment
        end_time = time.time()
        print(f"[AsyncRetriever] Fetch time: {end_time - start_time}s")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in fetch: {e}")
    
    return response



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
