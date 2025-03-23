from fastapi import FastAPI
from tasks.libs import install_libraries

app = FastAPI()

@app.post("/install_libraries")
def install_libraries(task: str):
    return install_libraries(task)
