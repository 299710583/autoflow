ARG BASE_IMAGE=kalilinux/kali-rolling
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/sources.list.d/google*.list /etc/apt/sources.list.d/chrome*.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        git \
        jq \
        python3 \
        python3-pip \
        ruby \
        dnsutils \
        whois \
        iputils-ping \
        traceroute \
        netcat-openbsd \
        openssl \
        nmap \
        whatweb \
        httpx-toolkit \
        wafw00f \
        nikto \
        sslscan \
        testssl.sh \
        nuclei \
        naabu \
        subfinder \
        dirb \
        dirsearch \
        gobuster \
        ffuf \
        feroxbuster \
        sqlmap \
        hydra \
        medusa \
        smbclient \
        enum4linux \
        smbmap \
        trivy \
        bandit \
        gitleaks \
    && python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed semgrep==1.124.0 \
    && rm -rf /var/lib/apt/lists/*

COPY nuclei-templates /opt/autoflow/nuclei-templates

WORKDIR /work

ENTRYPOINT []
