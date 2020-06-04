#!/bin/bash
set -e

if [ "$1" = "run" ]; then
  WORKER_CLASS=sync
  WORKER_CONNECTIONS=10
  MAX_REQUESTS_JITTER=10
  DJANGO_WSGI_MODULE=wsgi

  exec gunicorn ${DJANGO_WSGI_MODULE}:app \
    --worker-class=$WORKER_CLASS \
    --worker-connections=$WORKER_CONNECTIONS \
    --workers $WORKERS \
    --timeout $TIMEOUT \
    --max-requests $MAX_REQUESTS \
    --max-requests-jitter $MAX_REQUESTS_JITTER \
    --bind=0.0.0.0:$PORT
else
  exec "$@"
fi
