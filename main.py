from fastapi import FastAPI
from endpoints import router
from rag import initialize_rag

app = FastAPI()

# Include the router defined in endpoints.py
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    initialize_rag()

