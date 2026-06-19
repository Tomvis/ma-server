# syntax=docker/dockerfile:1

# Builder image. It builds the venv that will be copied to the final image
#
ARG BASE_IMAGE_VERSION=latest
FROM ghcr.io/music-assistant/base:$BASE_IMAGE_VERSION AS builder

ADD dist dist
COPY requirements_all.txt .

# ensure UV is installed
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# create venv which will be copied to the final image
ENV VIRTUAL_ENV=/app/venv
RUN uv venv $VIRTUAL_ENV

# Pre-install local sibling wheels so requirements_all.txt's pinned versions
# resolve without PyPI. On feature branches the server pins
# music-assistant-frontend / music-assistant-models / aiosendspin versions
# that aren't on PyPI yet (carry feature-branch-only commits — currently
# music-assistant-frontend@2.17.155, music-assistant-models@1.1.118 and
# aiosendspin@5.2.0.dev2+trackfields, which exposes extra track-info fields
# the rating-branch sendspin provider relies on). The matching wheels are
# dropped into dist/ alongside the server wheel before the docker build.
#
# requirements_all.txt is rewritten in-place to point aiosendspin at the
# local wheel — otherwise uv would resolve the bare ==5.2.0 pin from PyPI
# and overwrite our pre-installed trackfields build (PEP 440: 5.2.0.dev2 is
# a pre-release of 5.2.0, so the bare == constraint doesn't accept it).
RUN if ls dist/music_assistant_frontend-*.whl 2>/dev/null | grep -q .; then \
        uv pip install --no-cache dist/music_assistant_frontend-*.whl; \
    fi && \
    if ls dist/music_assistant_models-*.whl 2>/dev/null | grep -q .; then \
        uv pip install --no-cache dist/music_assistant_models-*.whl; \
    fi && \
    if ls dist/aiosendspin-*.whl 2>/dev/null | grep -q .; then \
        AIOSENDSPIN_WHL=$(ls dist/aiosendspin-*.whl | head -1); \
        uv pip install --no-cache "aiosendspin[server] @ file:${AIOSENDSPIN_WHL}"; \
        sed -i -E "s|^aiosendspin\[server\]==.*|aiosendspin[server] @ file:./${AIOSENDSPIN_WHL}|" requirements_all.txt; \
    fi

# pre-install ALL requirements into the venv
# comes at a cost of a slightly larger image size but is faster to start
# because we do not have to install dependencies at runtime
# --index-strategy: allow PyPI packages when also using the PyTorch extra index
# https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes
RUN uv pip install \
    --index-strategy unsafe-best-match \
    -r requirements_all.txt

# Install PyAV from pre-built wheel (built against system FFmpeg in base image)
# First verify the wheel version matches what pip resolved to avoid version mismatch
RUN REQUIRED_VERSION=$($VIRTUAL_ENV/bin/python -c "import importlib.metadata; print(importlib.metadata.version('av'))") && \
    WHEEL_VERSION=$(ls /usr/local/share/pyav-wheels/av*.whl | grep -oP 'av-\K[0-9.]+') && \
    if [ "$REQUIRED_VERSION" != "$WHEEL_VERSION" ]; then \
      echo "ERROR: PyAV version mismatch! Requirements need $REQUIRED_VERSION but base image has $WHEEL_VERSION" && \
      echo "Please rebuild the base image with the correct PyAV version." && \
      exit 1; \
    fi && \
    uv pip install --force-reinstall --no-deps /usr/local/share/pyav-wheels/av*.whl

# Install Music Assistant from prebuilt wheel
ARG MASS_VERSION
RUN uv pip install \
    --no-cache \
    "music-assistant@dist/music_assistant-${MASS_VERSION}-py3-none-any.whl"

# Pre-compile Python bytecode for faster startup
RUN $VIRTUAL_ENV/bin/python -m compileall -q $VIRTUAL_ENV/lib/python*/site-packages/music_assistant

# we need to set (very permissive) permissions to the workdir
# and /tmp to allow running the container as non-root
# IMPORTANT: chmod here, NOT on the final image, to avoid creating extra layers and increase size!
#
RUN chmod -R 777 /app

##################################################################################################

# FINAL docker image for music assistant server

FROM ghcr.io/music-assistant/base:$BASE_IMAGE_VERSION

ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# copy the already built /app dir
COPY --from=builder /app /app

# the /app contents have correct permissions but for some reason /app itself does not.
# so apply again, but ONLY to the dir (otherwise we increase the size)
RUN chmod 777 /app

# Set some labels
ARG MASS_VERSION
ARG TARGETPLATFORM
LABEL \
    org.opencontainers.image.title="Music Assistant Server" \
    org.opencontainers.image.description="Music Assistant is a free, opensource Media library manager that connects to your streaming services and a wide range of connected speakers. The server is the beating heart, the core of Music Assistant and must run on an always-on device like a Raspberry Pi, a NAS or an Intel NUC or alike." \
    org.opencontainers.image.source="https://github.com/music-assistant/server" \
    org.opencontainers.image.authors="The Music Assistant Team" \
    org.opencontainers.image.documentation="https://music-assistant.io" \
    org.opencontainers.image.licenses="Apache License 2.0" \
    io.hass.version="${MASS_VERSION}" \
    io.hass.type="addon" \
    io.hass.name="Music Assistant Server" \
    io.hass.description="Music Assistant Server" \
    io.hass.platform="${TARGETPLATFORM}" \
    io.hass.type="addon"

VOLUME [ "/data" ]
EXPOSE 8095

WORKDIR $VIRTUAL_ENV

# Entrypoint script that enables jemalloc for the main process only.
# MALLOC_CONF enables jemalloc's background thread so freed pages are returned to
# the OS while the process is idle; without it an idle server holds onto the peak
# RSS reached during startup (db migration, provider setup, metadata, image decode).
RUN printf '#!/bin/sh\n\
for path in /usr/lib/*/libjemalloc.so.2; do\n\
    [ -f "$path" ] && export LD_PRELOAD="$path" MALLOC_CONF="background_thread:true,dirty_decay_ms:5000,muzzy_decay_ms:5000" && break\n\
done\n\
exec mass "$@"\n' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh", "--data-dir", "/data", "--cache-dir", "/data/.cache"]
