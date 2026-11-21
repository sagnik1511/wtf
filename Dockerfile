# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE} AS builder
ARG BUILD_NATIVE=1
WORKDIR /build
RUN if [ "$BUILD_NATIVE" = "1" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends build-essential \
      && rm -rf /var/lib/apt/lists/* \
      && python -m pip install --no-cache-dir 'setuptools>=77' wheel; \
    elif [ "$BUILD_NATIVE" != "0" ]; then echo 'BUILD_NATIVE must be 0 or 1' >&2; exit 2; fi
COPY pyproject.toml LICENSE PYPI_README.md _build_support.py ./
COPY ecosystem/ ./ecosystem/
COPY work/build_native.py ./work/build_native.py
RUN if [ "$BUILD_NATIVE" = "1" ]; then python -m work.build_native; \
    elif [ "$BUILD_NATIVE" != "0" ]; then echo 'BUILD_NATIVE must be 0 or 1' >&2; exit 2; fi

FROM ${PYTHON_IMAGE} AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp
WORKDIR /app
COPY requirements.txt LICENSE ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 wtf \
    && mkdir -p /data/results \
    && chown -R wtf:wtf /data
COPY --from=builder /build/ecosystem/ ./ecosystem/
COPY viewer/ ./viewer/
COPY environments/ ./environments/
COPY maps/ ./maps/
COPY environment.schema.json ./
COPY examples/ ./examples/
USER wtf
EXPOSE 8000
ENTRYPOINT ["python", "-m", "ecosystem.cli"]
CMD ["environments/baseline.json", "--runs", "1", "--progress", "--output", "/data/results"]
