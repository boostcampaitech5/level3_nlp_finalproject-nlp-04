if __name__ == "__main__":
    import uvicorn

    uvicorn.run("Keyword Extraction.main:app", host="0.0.0.0", port=30008, reload=True)
