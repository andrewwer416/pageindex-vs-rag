# Deployment examples

These files show patterns for fronting the app with **Traefik** and a
**Cloudflare tunnel** so it serves on your own subdomain (e.g.
`pageindex.example.com`) without opening any ports.

Files in this directory are **examples**, not turnkey deploys. Edit
hostnames, network names, and tunnel IDs to match your environment.

## Two patterns

### A. Traefik using Docker labels (compose-driven)

Best when you already have a single-host Traefik that scans Docker
labels — typical in something like an AgentOps-style platform.

* `compose.traefik-labels.yaml.example` — app stack with Host-rule labels.

You'll need a Traefik instance already running on a shared external
network (`traefik-network` in the example), with the docker provider
enabled.

### B. Traefik file provider + Cloudflare tunnel (route-driven)

Best when Traefik routing lives in a `dynamic/` config dir and a
Cloudflare tunnel fronts the host. No public ports — Cloudflare's edge
holds the connection and forwards to Traefik.

* `compose.traefik-tunnel.yaml.example` — app stack joining a
  `tunnel-ingress` external network, no labels.
* `traefik-route.yaml.example` — drop this into your Traefik
  `dynamic/` dir; it adds the Host rule.
* `cloudflared-ingress-snippet.yaml.example` — append the ingress
  block to your existing `~/cloudflared/config.yml`, then
  `docker compose restart cloudflared` (or `systemctl restart
  cloudflared`).

A DNS record (CNAME `<your subdomain>` →
`<tunnel-id>.cfargotunnel.com`, proxied) must exist in Cloudflare for
the hostname to resolve.

## Where the indices live

The container mounts `./index` and `./data` from the project root. If
you're deploying from a non-default path, change the volume sources.

## A note on running the LLM somewhere else

The app talks to the LLM via `LLM_API_BASE`. If the LLM runs:

* **on the same host as the app** — set `LLM_API_BASE=http://host.docker.internal:11434`
* **on another host in a private network** (Tailscale, VPN, LAN) — use that host's IP directly
* **as a sibling container on the same compose network** — use the service name

The model server doesn't need to be reachable from the public internet.
