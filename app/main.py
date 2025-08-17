from fastapi import FastAPI

app = FastAPI(title="JAMJAM AI")


# 기본 루트 엔드포인트
@app.get("/")
def root():
    return {"message": "JAMJAM AI"}