from app import app
from waitress import serve
import config

if __name__ == "__main__":
    print(f"Starting live server on {config.FLASK_HOST}:{config.FLASK_PORT}...")
    serve(app, host=config.FLASK_HOST, port=config.FLASK_PORT)
