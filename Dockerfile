# cohmetrix
FROM ubuntu:noble

ENV TZ=America/Sao_Paulo
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-numpy \
    default-jre-headless \
    cmake \
    build-essential \
    libboost-all-dev \
    libeigen3-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    patch \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Use a venv so pip installs don't trip PEP 668's "externally-managed" guard.
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /opt/text_metrics

COPY . .

RUN pip3 install psycopg2-binary Cython

# --no-build-isolation: legacy pins in requirements.txt (and the vendored
# nlpnet fork below) don't declare PEP 517 build deps properly, so isolated
# builds either fail or compile C extensions against a different numpy than
# runtime, causing ABI mismatches. Reusing the current env (which already has
# numpy + Cython from the step above) keeps the build consistent.
RUN pip3 install --no-cache-dir --no-build-isolation -r requirements.txt

# Apply our Py3.12/modern-numpy patch to the vendored nlpnet fork.
RUN patch -d tools/nlpnet-py3 -p1 < patches/nlpnet-py3.patch

RUN pip3 install --no-build-isolation ./tools/nlpnet-py3

# punkt + punkt_tab: sentence/word tokenization (required by production).
# stopwords: only used by the dev preprocessing scripts in text_metrics/scripts/.
RUN python3 -m nltk.downloader punkt punkt_tab stopwords

# TODO: replace deprecated `setup.py install` with `pip3 install --no-build-isolation ./tools/idd3`
RUN cd tools/idd3 && python3 setup.py install

WORKDIR /opt/text_metrics
