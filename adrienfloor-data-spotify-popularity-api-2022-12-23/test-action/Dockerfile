FROM python:3.8.12-slim-buster
RUN apt-get update && apt-get install -y --no-install-recommends make
RUN pip install -U pip
COPY requirements.test.txt /requirements.test.txt
RUN pip install -r requirements.test.txt
COPY Makefile.test /Makefile.test
CMD make pytest -f /Makefile.test