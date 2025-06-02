from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router

app = FastAPI(
    title="RAG Prototype",
    description="RAG system with hardcoded input and local output",
    version="0.1.0",
)

# Allow all CORS (in case frontend gets added later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(api_router)

# Optional test root route
@app.get("/")
def read_root():
    return {"message": "RAG backend is running!"}
