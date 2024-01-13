# Build Stage
FROM node:14 AS build-env
WORKDIR /app
ARG VITE_APP_API_URL
ENV VITE_APP_API_URL=${VITE_APP_API_URL}

# Copy the frontend directory to the container
COPY . ./

# Install dependencies and build the project
RUN npm config set registry https://registry.npm.taobao.org/ \
    && npm install \
    && npm run build

# Run Stage
FROM nginx:alpine
# Copy the build output from the Build Stage to the Nginx web directory
COPY --from=build-env /app/dist /usr/share/nginx/html

# Optionally, you can copy a custom Nginx configuration file
# This is useful if you have specific Nginx settings you want to apply
COPY ./nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80 for HTTP
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
