// Local fixture for the wasm browser connection test. No external packages.
import http from 'node:http';
import crypto from 'node:crypto';

const sockets = new Map();
const requests = [];
const server = http.createServer((req, res) => {
  const origin = req.headers.origin;
  if (origin) res.setHeader('Access-Control-Allow-Origin', origin);
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  const url = new URL(req.url, 'http://127.0.0.1:18766');
  if (url.pathname === '/control/requests') {
    res.end(JSON.stringify(requests));
  } else if (url.pathname === '/control/close') {
    for (const [socket, path] of sockets) {
      if (path.includes(url.searchParams.get('channel'))) {
        socket.end(Buffer.from([0x88, 0]));
      }
    }
    res.end('closed');
  } else {
    requests.push({ url: req.url, cookie: req.headers.cookie });
    if (!req.headers.cookie?.includes('e2_session=test')) {
      res.writeHead(401);
      res.end('Session cookie required');
      return;
    }
    res.end(req.url);
  }
});
server.on('upgrade', (req, socket) => {
  requests.push({ url: req.url, cookie: req.headers.cookie });
  if (!req.headers.cookie?.includes('e2_session=test')) {
    socket.end('HTTP/1.1 401 Unauthorized\r\n\r\n');
    return;
  }
  const accept = crypto.createHash('sha1')
    .update(req.headers['sec-websocket-key'] + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11')
    .digest('base64');
  socket.write('HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ' + accept + '\r\n\r\n');
  sockets.set(socket, req.url);
  socket.on('error', () => sockets.delete(socket));
  socket.on('close', () => sockets.delete(socket));
  socket.on('data', () => {});
});
server.listen(18766, '127.0.0.1', () => console.log('Browser fixture listening on 18766'));
