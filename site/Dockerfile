# Static site image for Coolify / any Docker host.
# Build context: the site/ directory of the sparsify repo.
#   docker build -t sparsify-site ./site && docker run -p 8080:80 sparsify-site
FROM nginx:alpine

COPY . /usr/share/nginx/html

RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    gzip on;\n\
    gzip_types text/html text/css application/javascript image/svg+xml application/json;\n\
    add_header X-Content-Type-Options nosniff;\n\
    add_header Referrer-Policy strict-origin-when-cross-origin;\n\
    location / { try_files $uri $uri/ /index.html; }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80
