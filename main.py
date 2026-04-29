import uvicorn

from config import APP_HOST, APP_PORT, APP_RELOAD


def main():
    uvicorn.run("backend.app:app", host=APP_HOST, port=APP_PORT, reload=APP_RELOAD)


if __name__ == "__main__":
    main()
