# All done by: Sepehr
############################ BASE STAGE ###############################

FROM python:3.8-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0

############################ PRODUCTION STAGE ############################

FROM base as production

ENV BACKEND="/" \
    BACKEND_HOME="/Model" \
    BACKEND_MREG="/Model/model_registry"

COPY ./requirements.txt /app/requirements.txt
COPY ./server $BACKEND_HOME/server/
COPY ./model_registry $BACKEND_MREG
COPY ./model_pipeline.py  $BACKEND_HOME
COPY ./Database $BACKEND_HOME/Datasets

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

WORKDIR $BACKEND

CMD ["uvicorn", "Model.server.main:app", "--host", "0.0.0.0", "--port", "80"]


