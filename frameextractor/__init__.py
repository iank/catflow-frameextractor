from flask import Flask
from .config.settings import Config
from .routes import register_routes
import logging

def create_app():
    app = Flask(__name__)

    # Load config
    config = Config()

    # Set up logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    app.logger.setLevel(logging.INFO)


    # Add config attributes to Flask's app.config
    for key in vars(config):
        app.config[key] = getattr(config, key)

    register_routes(app)
    return app
