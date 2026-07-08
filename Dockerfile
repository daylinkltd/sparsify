# Static site image for Coolify / any Docker host.
#
# IMPORTANT: build with the REPO ROOT as context so install.sh can be
# served from the site root (Ollama-style `curl https://DOMAIN/install.sh | sh`):
#   docker build -t sparsify-site .
# In Coolify: Build Pack = Dockerfile, Dockerfile location = /Dockerfile,
# Base directory = / (repo root).
FROM nginx:alpine

COPY site/ /usr/share/nginx/html
COPY install.sh /usr/share/nginx/html/install.sh

RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    gzip on;\n\
    gzip_types text/html text/css application/javascript image/svg+xml application/json text/plain;\n\
    add_header X-Content-Type-Options nosniff;\n\
    add_header Referrer-Policy strict-origin-when-cross-origin;\n\
    location = /install.sh { default_type text/plain; }\n\
    location / { try_files $uri $uri/ /index.html; }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80
