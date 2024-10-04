# hash:sha256:03bd252eba2861b121fccd91ab7adf2ea329b125da55e44c45e674e21c71b64d
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.1.0-cuda11.8-mambaforge23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    aind-metadata-upgrader==0.0.20 \
    h5py==3.12.1 \
    pint==0.24.3 \
    roicat[all]==1.3.0

COPY postInstall /
RUN /postInstall
