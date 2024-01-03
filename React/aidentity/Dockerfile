############################ BASE STAGE ###############################
FROM node:lts-alpine AS base
WORKDIR /app
COPY package*.json ./
RUN npm install

########################### BUILDER STAGE #############################
FROM base AS builder
WORKDIR /app
COPY . .
RUN npm run build

########################### FINAL STAGE ##############################
FROM node:lts-alpine
WORKDIR /app
COPY --from=builder /app/build .
RUN npm install -g http-server
EXPOSE 8080
CMD ["http-server", "-p", "8080", "--cors"]
