if __name__ == "__main__":
    import uvicorn

    uvicorn.run("Sentimental.main:app", host="0.0.0.0", port=30007, reload=True)
