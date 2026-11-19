# Template only: replace the all-zero digest with the digest NVIDIA publishes
# for the JetPack/L4T image compatible with the measured Thor. The build helper
# rejects this placeholder and unpinned references.
ARG BASE_IMAGE=nvcr.io/nvidia/thor-jetpack-image:replace-with-tag@sha256:0000000000000000000000000000000000000000000000000000000000000000
FROM ${BASE_IMAGE}

ARG WORKER_UID=10001
ARG WORKER_GID=10001
ARG SOURCE_COMMIT=0000000000000000000000000000000000000000

LABEL org.opencontainers.image.revision=${SOURCE_COMMIT} \
      io.so101.thor-worker.mode="qualification-only"

WORKDIR /opt/so101-thor-worker

# No SO-101 package, robot/Feetech/serial/RealSense extra, or runtime package
# installation belongs here. PyTorch/CUDA/cuDNN/TensorRT come from the selected
# digest-pinned NVIDIA image; compatibility is measured rather than assumed.
COPY --chown=${WORKER_UID}:${WORKER_GID} scripts/ ./scripts/
COPY --chown=${WORKER_UID}:${WORKER_GID} configs/ ./configs/
COPY --chown=${WORKER_UID}:${WORKER_GID} schemas/ ./schemas/
COPY --chown=${WORKER_UID}:${WORKER_GID} requirements-worker.txt ./requirements-worker.txt
COPY --chown=${WORKER_UID}:${WORKER_GID} README.md ./README.md

RUN if command -v groupadd >/dev/null 2>&1; then \
      groupadd --gid "${WORKER_GID}" worker && \
      useradd --uid "${WORKER_UID}" --gid "${WORKER_GID}" \
        --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin worker; \
    else \
      addgroup -g "${WORKER_GID}" worker && \
      adduser -D -H -u "${WORKER_UID}" -G worker -s /sbin/nologin worker; \
    fi && \
    mkdir -p /input /output && \
    chown "${WORKER_UID}:${WORKER_GID}" /output

USER 10001:10001
ENV HOME=/nonexistent \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    THOR_WORKER_QUALIFICATION_ONLY=1 \
    THOR_WORKER_SOURCE_COMMIT=${SOURCE_COMMIT}
ENTRYPOINT ["python3", "/opt/so101-thor-worker/scripts/run_benchmarks.py"]
CMD ["--help"]
