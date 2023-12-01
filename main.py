from fastapi import FastAPI, UploadFile, File


app = FastAPI()


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    return {"filename": file.filename}
