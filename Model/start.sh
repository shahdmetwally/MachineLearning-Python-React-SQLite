#!/bin/bash

uvicorn Model.server.admin:app --host 0.0.0.0 --port 8000 &

uvicorn Model.server.user:app --host 0.0.0.0 --port 8001
