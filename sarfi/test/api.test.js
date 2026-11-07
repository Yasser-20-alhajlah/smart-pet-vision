'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const { openAuthDb, createUserStore } = require('../src/db');
const { createApp } = require('../src/app');

// dataDir null keeps every database in memory.
async function startServer(dataDir = null) {
  const db = openAuthDb(dataDir ? `${dataDir}/sarfi.db` : ':memory:');
  const users = createUserStore(dataDir ? `${dataDir}/users` : null);
  const server = http.createServer(createApp(db, users));
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const call = async (method, path, { token, body, raw, headers = {} } = {}) => {
    const h = { ...headers };
    if (token) h.Authorization = `Bearer ${token}`;
    let payload;
    if (raw !== undefined) {
      payload = raw;
      h['Content-Type'] = h['Content-Type'] || 'text/plain';
    } else if (body !== undefined) {
      payload = JSON.stringify(body);
      h['Content-Type'] = 'application/json';
    }
    const res = await fetch(base + path, { method, headers: h, body: payload });
    const text = await res.text();
    let json = null;
    try { json = JSON.parse(text); } catch { /* not json */ }
    return { status: res.status, json, text, headers: res.headers };
  };
  const close = () => new Promise((r) => server.close(() => { users.closeAll(); db.close(); r(); }));
  return { db, users, call, close };
}

async function withOwner() {
  const s = await startServer();
  const { code } = s.db.bootstrapOwnerInvite();
  const join = await s.call('POST', '/api/join', { body: { code, name: 'ياسر' } });
  assert.equal(join.status, 201);
  return { ...s, ownerToken: join.json.token };
}

const SMS = 'شراء عبر نقاط البيع مدى\nبطاقة: 1234*\nمبلغ: 45.50 ر.س\nلدى: STARBUCKS\nفي: 2026-11-05 14:30';

test('everything under /api requires a valid token', async () => {
  const s = await startServer();
  try {
    for (const [m, p] of [['GET', '/api/me'], ['GET', '/api/transactions'], ['POST', '/api/sms'], ['GET', '/api/summary'], ['POST', '/api/invites']]) {
      assert.equal((await s.call(m, p)).status, 401, `${m} ${p}`);
      assert.equal((await s.call(m, p, { token: 'wrong' })).status, 401, `${m} ${p} bad token`);
    }
  } finally {
    await s.close();
  }
});

test('owner code works once, then the owner is the only one who can invite', async () => {
  const s = await startServer();
  try {
    const { code } = s.db.bootstrapOwnerInvite();
    const first = await s.call('POST', '/api/join', { body: { code, name: 'ياسر' } });
    assert.equal(first.status, 201);
    assert.equal(first.json.user.role, 'owner');
    const again = await s.call('POST', '/api/join', { body: { code, name: 'متطفل' } });
    assert.equal(again.status, 403);
    assert.equal(s.db.bootstrapOwnerInvite(), null, 'no new owner code once a user exists');

    const inv = await s.call('POST', '/api/invites', { token: first.json.token, body: { note: 'أخوي' } });
    assert.equal(inv.status, 201);
    const member = await s.call('POST', '/api/join', { body: { code: inv.json.code, name: 'محمد' } });
    assert.equal(member.status, 201);
    assert.equal(member.json.user.role, 'member');

    // Members cannot invite or see members: access only goes through the owner.
    assert.equal((await s.call('POST', '/api/invites', { token: member.json.token })).status, 403);
    assert.equal((await s.call('GET', '/api/members', { token: member.json.token })).status, 403);
    assert.equal((await s.call('POST', '/api/join', { body: { code: inv.json.code, name: 'ثالث' } })).status, 403);
  } finally {
    await s.close();
  }
});

test('join with a made-up code fails, and repeated attempts are rate limited', async () => {
  const s = await startServer();
  try {
    let last;
    for (let i = 0; i < 11; i += 1) {
      last = await s.call('POST', '/api/join', { body: { code: `guess${i}`, name: 'x' } });
    }
    assert.equal(last.status, 429);
  } finally {
    await s.close();
  }
});

test('SMS from a Shortcut (plain text) becomes a transaction, duplicates are ignored', async () => {
  const s = await withOwner();
  try {
    const r = await s.call('POST', '/api/sms', { token: s.ownerToken, raw: SMS });
    assert.equal(r.status, 201);
    assert.equal(r.json.transaction.amount, 45.5);
    assert.equal(r.json.transaction.source, 'sms');
    const dup = await s.call('POST', '/api/sms', { token: s.ownerToken, body: { text: SMS } });
    assert.equal(dup.status, 200);
    assert.equal(dup.json.duplicate, true);
    const list = await s.call('GET', '/api/transactions?month=2026-11', { token: s.ownerToken });
    assert.equal(list.json.transactions.length, 1);
    const bad = await s.call('POST', '/api/sms', { token: s.ownerToken, raw: 'رمز التحقق 1234' });
    assert.equal(bad.status, 422);
  } finally {
    await s.close();
  }
});

test('users only ever see and change their own data', async () => {
  const s = await withOwner();
  try {
    const inv = await s.call('POST', '/api/invites', { token: s.ownerToken, body: {} });
    const member = (await s.call('POST', '/api/join', { body: { code: inv.json.code, name: 'محمد' } })).json.token;
    const own = await s.call('POST', '/api/sms', { token: s.ownerToken, raw: SMS });
    const id = own.json.transaction.id;

    const memberList = await s.call('GET', '/api/transactions?month=2026-11', { token: member });
    assert.equal(memberList.json.transactions.length, 0);
    assert.equal((await s.call('PATCH', `/api/transactions/${id}`, { token: member, body: { note: 'x' } })).status, 404);
    assert.equal((await s.call('DELETE', `/api/transactions/${id}`, { token: member })).status, 404);
    const sum = await s.call('GET', '/api/summary?month=2026-11', { token: member });
    assert.equal(sum.json.spent, 0);
  } finally {
    await s.close();
  }
});

test('manual entry, edit, summary by category, delete', async () => {
  const s = await withOwner();
  try {
    const t = s.ownerToken;
    await s.call('POST', '/api/sms', { token: t, raw: SMS });
    const manual = await s.call('POST', '/api/transactions', {
      token: t, body: { amount: 100, merchant: 'بنده', category: 'بقالة وتموين', occurred_at: '2026-11-03T10:00:00Z' },
    });
    assert.equal(manual.status, 201);
    await s.call('POST', '/api/transactions', { token: t, body: { amount: 5000, direction: 'in', occurred_at: '2026-11-01T06:00:00Z' } });

    const edited = await s.call('PATCH', `/api/transactions/${manual.json.transaction.id}`, { token: t, body: { category: 'تسوق', note: 'ملاحظة' } });
    assert.equal(edited.json.transaction.category, 'تسوق');
    assert.equal((await s.call('PATCH', `/api/transactions/${manual.json.transaction.id}`, { token: t, body: { category: 'مو موجود' } })).status, 400);

    const sum = await s.call('GET', '/api/summary?month=2026-11', { token: t });
    assert.equal(sum.json.spent, 145.5);
    assert.equal(sum.json.income, 5000);
    assert.equal(sum.json.net, 4854.5);
    assert.deepEqual(sum.json.byCategory.map((c) => c.category), ['تسوق', 'مطاعم وقهوة']);

    assert.equal((await s.call('DELETE', `/api/transactions/${manual.json.transaction.id}`, { token: t })).status, 200);
    assert.equal((await s.call('GET', '/api/summary?month=2026-11', { token: t })).json.spent, 45.5);
  } finally {
    await s.close();
  }
});

test('month boundaries use Riyadh time', async () => {
  const s = await withOwner();
  try {
    // 22:30 UTC on Oct 31 is 01:30 on Nov 1 in Riyadh.
    await s.call('POST', '/api/transactions', { token: s.ownerToken, body: { amount: 10, occurred_at: '2026-10-31T22:30:00Z' } });
    assert.equal((await s.call('GET', '/api/summary?month=2026-11', { token: s.ownerToken })).json.spent, 10);
    assert.equal((await s.call('GET', '/api/summary?month=2026-10', { token: s.ownerToken })).json.spent, 0);
    assert.equal((await s.call('GET', '/api/summary?month=2026-13', { token: s.ownerToken })).status, 400);
  } finally {
    await s.close();
  }
});

test('owner can remove a member; their token and data are gone', async () => {
  const s = await withOwner();
  try {
    const inv = await s.call('POST', '/api/invites', { token: s.ownerToken, body: {} });
    const joined = (await s.call('POST', '/api/join', { body: { code: inv.json.code, name: 'محمد' } })).json;
    await s.call('POST', '/api/sms', { token: joined.token, raw: SMS });
    assert.equal((await s.call('DELETE', `/api/members/${joined.user.id}`, { token: s.ownerToken })).status, 200);
    assert.equal((await s.call('GET', '/api/me', { token: joined.token })).status, 401);
    const members = await s.call('GET', '/api/members', { token: s.ownerToken });
    assert.equal(members.json.members.length, 1);
  } finally {
    await s.close();
  }
});

test('rotating the token invalidates the old one', async () => {
  const s = await withOwner();
  try {
    const r = await s.call('POST', '/api/me/token', { token: s.ownerToken });
    assert.equal((await s.call('GET', '/api/me', { token: s.ownerToken })).status, 401);
    assert.equal((await s.call('GET', '/api/me', { token: r.json.token })).status, 200);
  } finally {
    await s.close();
  }
});

test('CSV export neutralises spreadsheet formulas', async () => {
  const s = await withOwner();
  try {
    await s.call('POST', '/api/transactions', { token: s.ownerToken, body: { amount: 1, merchant: '=HYPERLINK("x")', occurred_at: '2026-11-02T10:00:00Z' } });
    const r = await s.call('GET', '/api/export.csv?month=2026-11', { token: s.ownerToken });
    assert.equal(r.status, 200);
    assert.match(r.text, /"'=HYPERLINK\(""x""\)"/);
  } finally {
    await s.close();
  }
});

test('oversized bodies, bad JSON and path traversal are rejected', async () => {
  const s = await withOwner();
  try {
    assert.equal((await s.call('POST', '/api/sms', { token: s.ownerToken, raw: 'x'.repeat(20000) })).status, 413);
    assert.equal((await s.call('POST', '/api/transactions', { token: s.ownerToken, raw: '{bad', headers: { 'Content-Type': 'application/json' } })).status, 400);
    assert.equal((await s.call('GET', '/..%2f..%2fsrc%2fdb.js')).status, 404);
    const page = await s.call('GET', '/');
    assert.equal(page.status, 200);
    assert.match(page.headers.get('content-security-policy'), /default-src 'self'/);
  } finally {
    await s.close();
  }
});

test('each user gets their own database file, deleted when the member is removed', async () => {
  const fs = require('node:fs');
  const os = require('node:os');
  const path = require('node:path');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'sarfi-'));
  const s = await startServer(dir);
  try {
    const { code } = s.db.bootstrapOwnerInvite();
    const owner = (await s.call('POST', '/api/join', { body: { code, name: 'ياسر' } })).json;
    const inv = await s.call('POST', '/api/invites', { token: owner.token, body: {} });
    const member = (await s.call('POST', '/api/join', { body: { code: inv.json.code, name: 'محمد' } })).json;
    await s.call('POST', '/api/sms', { token: owner.token, raw: SMS });
    await s.call('POST', '/api/sms', { token: member.token, raw: SMS });

    const ownerFile = s.users.fileFor(owner.user.id);
    const memberFile = s.users.fileFor(member.user.id);
    assert.notEqual(ownerFile, memberFile);
    assert.ok(fs.existsSync(ownerFile) && fs.existsSync(memberFile));

    // The central database holds no financial data.
    const { DatabaseSync } = require('node:sqlite');
    const central = new DatabaseSync(path.join(dir, 'sarfi.db'), { readOnly: true });
    const tables = central.prepare("SELECT name FROM sqlite_master WHERE type = 'table'").all().map((r) => r.name);
    central.close();
    assert.deepEqual(tables.sort(), ['invites', 'users']);

    assert.equal((await s.call('DELETE', `/api/members/${member.user.id}`, { token: owner.token })).status, 200);
    assert.equal(fs.existsSync(memberFile), false);
    assert.equal(fs.existsSync(ownerFile), true);
  } finally {
    await s.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('settings: validated, merged, and private per user', async () => {
  const s = await withOwner();
  try {
    const t = s.ownerToken;
    const ok = await s.call('PUT', '/api/settings', { token: t, body: { salary: 12000, payday: 27, savings_pct: 25 } });
    assert.equal(ok.status, 200);
    assert.deepEqual(ok.json.settings, { salary: 12000, payday: 27, savings_pct: 25, budget_overrides: {} });

    const merged = await s.call('PUT', '/api/settings', { token: t, body: { budget_overrides: { 'مطاعم وقهوة': 800 } } });
    assert.equal(merged.json.settings.salary, 12000);
    assert.equal(merged.json.settings.budget_overrides['مطاعم وقهوة'], 800);
    const cleared = await s.call('PUT', '/api/settings', { token: t, body: { salary: null, budget_overrides: { 'مطاعم وقهوة': null } } });
    assert.equal('salary' in cleared.json.settings, false);
    assert.deepEqual(cleared.json.settings.budget_overrides, {});

    for (const body of [{ salary: -5 }, { salary: 'abc' }, { payday: 0 }, { payday: 32 }, { payday: 2.5 },
      { savings_pct: 90 }, { budget_overrides: { 'غير موجود': 10 } }, { budget_overrides: [1] }]) {
      assert.equal((await s.call('PUT', '/api/settings', { token: t, body })).status, 400, JSON.stringify(body));
    }

    const inv = await s.call('POST', '/api/invites', { token: t, body: {} });
    const member = (await s.call('POST', '/api/join', { body: { code: inv.json.code, name: 'محمد' } })).json.token;
    await s.call('PUT', '/api/settings', { token: t, body: { salary: 12000 } });
    assert.deepEqual((await s.call('GET', '/api/settings', { token: member })).json.settings, {});
    assert.equal((await s.call('GET', '/api/settings')).status, 401);
  } finally {
    await s.close();
  }
});

test('advisor endpoint returns a plan from the user salary and only their data', async () => {
  const s = await withOwner();
  try {
    const t = s.ownerToken;
    const none = await s.call('GET', '/api/advisor', { token: t });
    assert.equal(none.status, 200);
    assert.equal(none.json.plan, null);
    assert.equal(none.json.insights[0].level, 'alert');

    await s.call('PUT', '/api/settings', { token: t, body: { salary: 10000, payday: 27 } });
    await s.call('POST', '/api/transactions', { token: t, body: { amount: 300, category: 'مطاعم وقهوة' } });
    const r = await s.call('GET', '/api/advisor', { token: t });
    assert.equal(r.json.plan.salary, 10000);
    assert.equal(r.json.plan.savings.amount, 2000);
    assert.equal(r.json.progress.spent, 300);

    const inv = await s.call('POST', '/api/invites', { token: t, body: {} });
    const member = (await s.call('POST', '/api/join', { body: { code: inv.json.code, name: 'محمد' } })).json.token;
    const m = await s.call('GET', '/api/advisor', { token: member });
    assert.equal(m.json.plan, null);
    assert.equal(m.json.progress.spent, 0);
    assert.equal((await s.call('GET', '/api/advisor')).status, 401);
  } finally {
    await s.close();
  }
});
