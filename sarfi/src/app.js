'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { parseSms, CATEGORY_NAMES } = require('./parser');

const PUBLIC_DIR = path.join(__dirname, '..', 'public');
const MAX_BODY = 16 * 1024;
const JOIN_LIMIT = { max: 10, windowMs: 15 * 60 * 1000 };

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.webmanifest': 'application/manifest+json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
};

const SECURITY_HEADERS = {
  'Content-Security-Policy': "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'no-referrer',
  'X-Frame-Options': 'DENY',
};

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function send(res, status, body, headers = {}) {
  const isJson = typeof body !== 'string' && !Buffer.isBuffer(body);
  const payload = isJson ? JSON.stringify(body) : body;
  res.writeHead(status, {
    ...SECURITY_HEADERS,
    'Content-Type': isJson ? 'application/json; charset=utf-8' : 'text/plain; charset=utf-8',
    'Cache-Control': 'no-store',
    ...headers,
  });
  res.end(payload);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let size = 0;
    let tooLarge = false;
    const chunks = [];
    req.on('data', (c) => {
      if (tooLarge) return; // keep draining so the 413 response can be sent
      size += c.length;
      if (size > MAX_BODY) {
        tooLarge = true;
        chunks.length = 0;
        reject(new HttpError(413, 'الطلب كبير جداً'));
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

async function readJson(req) {
  const raw = await readBody(req);
  if (!raw) return {};
  try {
    const v = JSON.parse(raw);
    if (v && typeof v === 'object' && !Array.isArray(v)) return v;
  } catch {
    // fall through
  }
  throw new HttpError(400, 'صيغة JSON غير صحيحة');
}

const MONTH_RE = /^\d{4}-(0[1-9]|1[0-2])$/;
function currentMonth() {
  // Riyadh is UTC+3 with no DST.
  return new Date(Date.now() + 3 * 3600 * 1000).toISOString().slice(0, 7);
}
function monthParam(url) {
  const m = url.searchParams.get('month') || currentMonth();
  if (!MONTH_RE.test(m)) throw new HttpError(400, 'الشهر لازم يكون بصيغة YYYY-MM');
  return m;
}

function cleanText(v, max) {
  if (v === undefined || v === null) return null;
  if (typeof v !== 'string') throw new HttpError(400, 'قيمة نصية غير صحيحة');
  const s = v.replace(/[\u0000-\u001f\u007f]/g, ' ').trim();
  return s ? s.slice(0, max) : null;
}

function validateManual(body) {
  const amount = Number(body.amount);
  if (!Number.isFinite(amount) || amount <= 0 || amount > 10_000_000) throw new HttpError(400, 'المبلغ غير صحيح');
  const direction = body.direction === 'in' ? 'in' : 'out';
  const category = direction === 'in' ? 'دخل' : (cleanText(body.category, 40) || 'أخرى');
  if (!CATEGORY_NAMES.includes(category)) throw new HttpError(400, 'التصنيف غير معروف');
  let occurred = new Date();
  if (body.occurred_at) {
    occurred = new Date(body.occurred_at);
    if (Number.isNaN(occurred.getTime())) throw new HttpError(400, 'التاريخ غير صحيح');
  }
  return {
    amount: Math.round(amount * 100) / 100,
    direction,
    type: direction === 'in' ? 'incoming' : 'purchase',
    merchant: cleanText(body.merchant, 80),
    category,
    note: cleanText(body.note, 200),
    occurred_at: occurred.toISOString(),
  };
}

function csvCell(v) {
  const s = v === null || v === undefined ? '' : String(v);
  // Neutralise spreadsheet formulas and quote everything.
  const safe = /^[=+\-@\t\r]/.test(s) ? `'${s}` : s;
  return `"${safe.replace(/"/g, '""')}"`;
}

function createApp(db, { log = () => {}, trustProxy = false } = {}) {
  const joinAttempts = new Map();

  function clientIp(req) {
    if (trustProxy) {
      const fwd = req.headers['x-forwarded-for'];
      if (fwd) return String(fwd).split(',')[0].trim();
    }
    return req.socket.remoteAddress || 'unknown';
  }

  function checkJoinRate(ip) {
    const now = Date.now();
    const entry = joinAttempts.get(ip);
    if (!entry || now - entry.start > JOIN_LIMIT.windowMs) {
      joinAttempts.set(ip, { start: now, count: 1 });
      return;
    }
    entry.count += 1;
    if (entry.count > JOIN_LIMIT.max) throw new HttpError(429, 'محاولات كثيرة، جرّب بعد شوي');
  }

  function auth(req) {
    const h = req.headers.authorization || '';
    const token = h.startsWith('Bearer ') ? h.slice(7).trim() : null;
    const user = db.userByToken(token);
    if (!user) throw new HttpError(401, 'غير مصرح');
    return user;
  }

  function requireOwner(user) {
    if (user.role !== 'owner') throw new HttpError(403, 'هذي الخاصية لصاحب التطبيق فقط');
  }

  function idParam(match) {
    const id = Number(match[1]);
    if (!Number.isSafeInteger(id) || id <= 0) throw new HttpError(400, 'رقم غير صحيح');
    return id;
  }

  const routes = [
    ['POST', /^\/api\/join$/, async (req) => {
      checkJoinRate(clientIp(req));
      const body = await readJson(req);
      const name = cleanText(body.name, 40);
      if (!name) throw new HttpError(400, 'اكتب اسمك');
      const result = db.redeemInvite(cleanText(body.code, 64) || '', name);
      if (!result) throw new HttpError(403, 'رمز الدعوة غير صحيح أو مستخدم أو منتهي');
      log(`user joined: id=${result.user.id} role=${result.user.role}`);
      return [201, result];
    }],

    ['GET', /^\/api\/me$/, async (req) => [200, { user: auth(req), categories: CATEGORY_NAMES }]],

    ['POST', /^\/api\/me\/token$/, async (req) => {
      const user = auth(req);
      return [200, { token: db.rotateToken(user.id) }];
    }],

    // Accepts JSON {"text": "..."} or a raw text/plain body, which is the
    // simplest thing to send from an iOS Shortcut.
    ['POST', /^\/api\/sms$/, async (req) => {
      const user = auth(req);
      const ctype = String(req.headers['content-type'] || '');
      let text;
      if (ctype.includes('application/json')) {
        const body = await readJson(req);
        text = body.text;
      } else {
        text = await readBody(req);
      }
      const parsed = parseSms(text);
      if (!parsed.ok) throw new HttpError(422, parsed.error);
      const result = db.addTransaction(user.id, parsed.transaction, 'sms');
      return [result.duplicate ? 200 : 201, result];
    }],

    ['POST', /^\/api\/transactions$/, async (req) => {
      const user = auth(req);
      const tx = validateManual(await readJson(req));
      return [201, db.addTransaction(user.id, tx, 'manual')];
    }],

    ['GET', /^\/api\/transactions$/, async (req, url) => {
      const user = auth(req);
      return [200, { transactions: db.transactions(user.id, monthParam(url)) }];
    }],

    ['PATCH', /^\/api\/transactions\/(\d+)$/, async (req, url, m) => {
      const user = auth(req);
      const body = await readJson(req);
      const fields = {};
      if (body.category !== undefined) {
        const c = cleanText(body.category, 40);
        if (!CATEGORY_NAMES.includes(c)) throw new HttpError(400, 'التصنيف غير معروف');
        fields.category = c;
      }
      if (body.merchant !== undefined) fields.merchant = cleanText(body.merchant, 80);
      if (body.note !== undefined) fields.note = cleanText(body.note, 200);
      const tx = db.updateTransaction(user.id, idParam(m), fields);
      if (!tx) throw new HttpError(404, 'العملية غير موجودة');
      return [200, { transaction: tx }];
    }],

    ['DELETE', /^\/api\/transactions\/(\d+)$/, async (req, url, m) => {
      const user = auth(req);
      if (!db.deleteTransaction(user.id, idParam(m))) throw new HttpError(404, 'العملية غير موجودة');
      return [200, { ok: true }];
    }],

    ['GET', /^\/api\/summary$/, async (req, url) => {
      const user = auth(req);
      return [200, db.summary(user.id, monthParam(url))];
    }],

    ['GET', /^\/api\/export\.csv$/, async (req, url) => {
      const user = auth(req);
      const month = monthParam(url);
      const header = ['التاريخ', 'النوع', 'المبلغ', 'التاجر', 'التصنيف', 'البطاقة', 'ملاحظة', 'المصدر'];
      const lines = [header.map(csvCell).join(',')];
      for (const t of db.transactions(user.id, month)) {
        lines.push([t.occurred_at, t.direction === 'in' ? 'دخل' : 'صرف', t.amount, t.merchant,
          t.category, t.card_last4, t.note, t.source].map(csvCell).join(','));
      }
      return [200, `﻿${lines.join('\r\n')}`, {
        'Content-Type': 'text/csv; charset=utf-8',
        'Content-Disposition': `attachment; filename="sarfi-${month}.csv"`,
      }];
    }],

    ['POST', /^\/api\/invites$/, async (req) => {
      const user = auth(req);
      requireOwner(user);
      const body = await readJson(req);
      return [201, db.createMemberInvite(user.id, cleanText(body.note, 60))];
    }],

    ['GET', /^\/api\/invites$/, async (req) => {
      requireOwner(auth(req));
      return [200, { invites: db.pendingInvites() }];
    }],

    ['DELETE', /^\/api\/invites\/(\d+)$/, async (req, url, m) => {
      requireOwner(auth(req));
      if (!db.revokeInvite(idParam(m))) throw new HttpError(404, 'الدعوة غير موجودة');
      return [200, { ok: true }];
    }],

    ['GET', /^\/api\/members$/, async (req) => {
      requireOwner(auth(req));
      return [200, { members: db.members() }];
    }],

    ['DELETE', /^\/api\/members\/(\d+)$/, async (req, url, m) => {
      requireOwner(auth(req));
      if (!db.deleteMember(idParam(m))) throw new HttpError(404, 'العضو غير موجود');
      return [200, { ok: true }];
    }],
  ];

  function serveStatic(req, res, url) {
    if (req.method !== 'GET' && req.method !== 'HEAD') return send(res, 405, 'Method Not Allowed');
    let rel = decodeURIComponent(url.pathname);
    if (rel === '/' || !path.extname(rel)) rel = '/index.html';
    const file = path.normalize(path.join(PUBLIC_DIR, rel));
    if (!file.startsWith(PUBLIC_DIR + path.sep)) return send(res, 404, 'Not Found');
    fs.readFile(file, (err, data) => {
      if (err) return send(res, 404, 'Not Found');
      const headers = { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' };
      if (path.basename(file) === 'sw.js') headers['Service-Worker-Allowed'] = '/';
      headers['Cache-Control'] = 'no-cache';
      return send(res, 200, data, headers);
    });
    return undefined;
  }

  return async function handler(req, res) {
    let url;
    try {
      url = new URL(req.url, 'http://localhost');
    } catch {
      return send(res, 400, { error: 'رابط غير صحيح' });
    }
    if (!url.pathname.startsWith('/api/')) return serveStatic(req, res, url);

    try {
      for (const [method, re, fn] of routes) {
        const m = re.exec(url.pathname);
        if (m && method === req.method) {
          const [status, body, headers] = await fn(req, url, m);
          return send(res, status, body, headers);
        }
      }
      return send(res, 404, { error: 'غير موجود' });
    } catch (err) {
      if (err instanceof HttpError) return send(res, err.status, { error: err.message });
      log(`error: ${err.stack || err}`);
      return send(res, 500, { error: 'خطأ في الخادم' });
    }
  };
}

module.exports = { createApp };
