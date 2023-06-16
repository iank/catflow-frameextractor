# catflow-frameextractor
Video preannotation and filter for an object recognition data pipeline

# Set up

# Setup

* Install [pre-commit](https://pre-commit.com/#install) in your virtualenv. Run
`pre-commit install` after cloning this repository.

* Install dependencies:

```
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

# Develop

```
pip install --editable .
```

# Build

```
python setup.py sdist bdist_wheel
```

# Install

```
pip install dist/frameextractor-*-py3-none-any.whl
```

# Docker

```
docker build -t iank1/frameextractor:latest .
```

See [catflow-docker](https://github.com/iank/catflow-docker) for `docker-compose.yml`
