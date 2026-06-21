import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const publicDir = join(__dirname, "public");

const PORT = Number(process.env.FRONTEND_PORT || 3000);
const API_BASE = process.env.EKGA_API_BASE || "http://127.0.0.1:8081";

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

const server = createServer(async (req, res) => {
  try {
    if (req.url?.startsWith("/api/")) {
      await proxyToBackend(req, res);
      return;
    }

    await serveStatic(req.url || "/", res);
  } catch (error) {
    console.error(error);
    sendJson(res, 500, { error: "Frontend server error" });
  }
});

server.listen(PORT, () => {
  console.log(`EKGA frontend running at http://127.0.0.1:${PORT}`);
  console.log(`Proxying API requests to ${API_BASE}`);
});

async function proxyToBackend(req, res) {
  const targetPath = req.url.replace(/^\/api/, "");
  const body = await readBody(req);
  const headers = { "content-type": req.headers["content-type"] || "application/json" };

  const response = await fetch(`${API_BASE}${targetPath}`, {
    method: req.method,
    headers,
    body: ["GET", "HEAD"].includes(req.method || "GET") ? undefined : body,
  });

  const text = await response.text();
  res.writeHead(response.status, {
    "content-type": response.headers.get("content-type") || "application/json; charset=utf-8",
  });
  res.end(text);
}

async function serveStatic(url, res) {
  const cleanUrl = url.split("?")[0];
  const requestedPath = cleanUrl === "/" ? "/index.html" : cleanUrl;
  const safePath = normalize(requestedPath).replace(/^(\.\.[/\\])+/, "");
  const filePath = join(publicDir, safePath);
  const ext = extname(filePath);

  try {
    const file = await readFile(filePath);
    res.writeHead(200, {
      "content-type": contentTypes[ext] || "application/octet-stream",
      "cache-control": "no-store",
    });
    res.end(file);
  } catch {
    const fallback = await readFile(join(publicDir, "index.html"));
    res.writeHead(200, { "content-type": contentTypes[".html"], "cache-control": "no-store" });
    res.end(fallback);
  }
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (chunk) => chunks.push(chunk));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

function sendJson(res, status, payload) {
  res.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(payload));
}
