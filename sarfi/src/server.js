'use strict';

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { openDb } = require('./db');
const { createApp } = require('./app');

const PORT = Number(process.env.PORT) || 3000;
const HOST = process.env.HOST || '0.0.0.0';
const DATA_DIR = process.env.DATA_DIR || path.join(__dirname, '..', 'data');
const TRUST_PROXY = process.env.TRUST_PROXY === '1';

fs.mkdirSync(DATA_DIR, { recursive: true, mode: 0o700 });
const db = openDb(path.join(DATA_DIR, 'sarfi.db'));

const log = (msg) => console.log(`[${new Date().toISOString()}] ${msg}`);
const server = http.createServer(createApp(db, { log, trustProxy: TRUST_PROXY }));

server.listen(PORT, HOST, () => {
  log(`صرفي يعمل على http://localhost:${PORT}`);
  const invite = db.bootstrapOwnerInvite();
  if (invite) {
    log('ما فيه مستخدمين بعد. رمز دعوة صاحب التطبيق (استخدام مرة واحدة):');
    console.log(`\n    ${invite.code}\n`);
    log('افتح التطبيق، واختر "انضمام"، وحط هذا الرمز. أول شخص يستخدمه يصير صاحب التطبيق.');
  }
});

function shutdown() {
  server.close(() => {
    db.close();
    process.exit(0);
  });
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
