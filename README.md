# catflow-frameextractor
Video preannotation and filter for an object recognition data pipeline

# Build
```
python setup.py sdist bdist_wheel
```

# Install
```
pip install dist/frameextractor-0.4.2-py3-none-any.whl -f https://download.pytorch.org/whl/torch_stable.html
```

# Docker

```
docker build -t iank1/frameextractor:v0.4.2 .
```

See [catflow-docker](https://github.com/iank/catflow-docker) for `docker-compose.yml`
