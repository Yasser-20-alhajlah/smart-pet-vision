'use strict';

// Storage is split in two:
// - one central auth database (users, invites) that holds no financial data;
// - one separate database file per user for that user's transactions and
//   settings. Removing a member deletes their file.

const fs = require('node:fs');
const path = require('node:path');
const { DatabaseSync } = require('node:sqlite');
const crypto = require('node:crypto');

const INVITE_TTL_MS = 7 * 24 * 60 * 60 * 1000;
const DUPLICATE_WINDOW_MS = 2 * 60 * 1000;
// Months and days are grouped in Riyadh time (UTC+3, no DST).
const RIYADH = "'+3 hours'";

const sha256 = (s) => crypto.createHash('sha256').update(s).digest('hex');
const randomToken = (bytes) => crypto.randomBytes(bytes).toString('base64url');
const nowIso = () => new Date().toISOString();

function openAuthDb(file) {
  const db = new DatabaseSync(file);
  db.exec(`
    PRAGMA journal_mode = WAL;
    PRAGMA foreign_keys = ON;
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL,
      role TEXT NOT NULL CHECK (role IN ('owner', 'member')),
      token_hash TEXT NOT NULL UNIQUE,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS invites (
      id INTEGER PRIMARY KEY,
      code_hash TEXT NOT NULL UNIQUE,
      role TEXT NOT NULL CHECK (role IN ('owner', 'member')),
      note TEXT,
      created_by INTEGER REFERENCES users(id) ON DELETE CASCADE,
      used_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
      created_at TEXT NOT NULL,
      expires_at TEXT NOT NULL,
      used_at TEXT
    );
  `);

  const q = {
    userCount: db.prepare('SELECT COUNT(*) AS n FROM users'),
    userByToken: db.prepare('SELECT id, name, role, created_at FROM users WHERE token_hash = ?'),
    insertUser: db.prepare('INSERT INTO users (name, role, token_hash, created_at) VALUES (?, ?, ?, ?)'),
    setToken: db.prepare('UPDATE users SET token_hash = ? WHERE id = ?'),
    members: db.prepare('SELECT id, name, role, created_at FROM users ORDER BY id'),
    deleteMember: db.prepare("DELETE FROM users WHERE id = ? AND role = 'member'"),
    insertInvite: db.prepare(`INSERT INTO invites (code_hash, role, note, created_by, created_at, expires_at)
      VALUES (?, ?, ?, ?, ?, ?)`),
    inviteByHash: db.prepare('SELECT * FROM invites WHERE code_hash = ?'),
    useInvite: db.prepare('UPDATE invites SET used_by = ?, used_at = ? WHERE id = ? AND used_at IS NULL'),
    dropOwnerInvites: db.prepare("DELETE FROM invites WHERE role = 'owner' AND used_at IS NULL"),
    pendingInvites: db.prepare(`SELECT id, note, created_at, expires_at FROM invites
      WHERE role = 'member' AND used_at IS NULL AND expires_at > ? ORDER BY id DESC`),
    revokeInvite: db.prepare("DELETE FROM invites WHERE id = ? AND role = 'member' AND used_at IS NULL"),
  };

  function createInvite(role, createdBy, note) {
    const code = randomToken(12);
    const now = Date.now();
    q.insertInvite.run(sha256(code), role, note || null, createdBy, new Date(now).toISOString(),
      new Date(now + INVITE_TTL_MS).toISOString());
    return { code, expires_at: new Date(now + INVITE_TTL_MS).toISOString() };
  }

  return {
    close: () => db.close(),
    hasUsers: () => q.userCount.get().n > 0,

    // With no users yet, issue a fresh one-time owner code (old ones are dropped
    // because only their hashes are stored and they cannot be shown again).
    bootstrapOwnerInvite() {
      if (q.userCount.get().n > 0) return null;
      q.dropOwnerInvites.run();
      return createInvite('owner', null, 'owner');
    },

    createMemberInvite: (ownerId, note) => createInvite('member', ownerId, note),
    pendingInvites: () => q.pendingInvites.all(nowIso()),
    revokeInvite: (id) => q.revokeInvite.run(id).changes > 0,

    // Redeems a single-use code inside a transaction so two concurrent joins
    // cannot both use it.
    redeemInvite(code, name) {
      const invite = q.inviteByHash.get(sha256(String(code)));
      if (!invite || invite.used_at || invite.expires_at <= nowIso()) return null;
      const token = randomToken(32);
      db.exec('BEGIN IMMEDIATE');
      try {
        const info = q.insertUser.run(name, invite.role, sha256(token), nowIso());
        const userId = Number(info.lastInsertRowid);
        if (q.useInvite.run(userId, nowIso(), invite.id).changes !== 1) throw new Error('used');
        db.exec('COMMIT');
        return { token, user: { id: userId, name, role: invite.role } };
      } catch (err) {
        db.exec('ROLLBACK');
        if (err.message === 'used') return null;
        throw err;
      }
    },

    userByToken: (token) => (token ? q.userByToken.get(sha256(token)) || null : null),
    rotateToken(userId) {
      const token = randomToken(32);
      q.setToken.run(sha256(token), userId);
      return token;
    },
    members: () => q.members.all(),
    deleteMember: (id) => q.deleteMember.run(id).changes > 0,
  };
}

function openUserDb(file) {
  const db = new DatabaseSync(file);
  db.exec(`
    PRAGMA journal_mode = WAL;
    CREATE TABLE IF NOT EXISTS transactions (
      id INTEGER PRIMARY KEY,
      amount REAL NOT NULL CHECK (amount > 0),
      direction TEXT NOT NULL CHECK (direction IN ('in', 'out')),
      type TEXT NOT NULL,
      merchant TEXT,
      category TEXT NOT NULL,
      card_last4 TEXT,
      note TEXT,
      source TEXT NOT NULL CHECK (source IN ('sms', 'manual')),
      raw_text TEXT,
      occurred_at TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS tx_time ON transactions(occurred_at);
    CREATE TABLE IF NOT EXISTS settings (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
  `);

  const COLS = 'id, amount, direction, type, merchant, category, card_last4, note, source, occurred_at';
  const q = {
    insertTx: db.prepare(`INSERT INTO transactions
      (amount, direction, type, merchant, category, card_last4, note, source, raw_text, occurred_at, created_at)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`),
    recentDuplicate: db.prepare('SELECT id FROM transactions WHERE raw_text = ? AND created_at > ? LIMIT 1'),
    txById: db.prepare('SELECT * FROM transactions WHERE id = ?'),
    txByMonth: db.prepare(`SELECT ${COLS} FROM transactions
      WHERE strftime('%Y-%m', occurred_at, ${RIYADH}) = ? ORDER BY occurred_at DESC, id DESC`),
    txSince: db.prepare(`SELECT ${COLS} FROM transactions WHERE occurred_at >= ? ORDER BY occurred_at`),
    txCount: db.prepare('SELECT COUNT(*) AS n FROM transactions'),
    updateTx: db.prepare('UPDATE transactions SET category = ?, merchant = ?, note = ? WHERE id = ?'),
    deleteTx: db.prepare('DELETE FROM transactions WHERE id = ?'),
    totals: db.prepare(`SELECT direction, ROUND(SUM(amount), 2) AS total, COUNT(*) AS n FROM transactions
      WHERE strftime('%Y-%m', occurred_at, ${RIYADH}) = ? GROUP BY direction`),
    byCategory: db.prepare(`SELECT category, ROUND(SUM(amount), 2) AS total, COUNT(*) AS n FROM transactions
      WHERE direction = 'out' AND strftime('%Y-%m', occurred_at, ${RIYADH}) = ?
      GROUP BY category ORDER BY total DESC`),
    byDay: db.prepare(`SELECT strftime('%d', occurred_at, ${RIYADH}) AS day, ROUND(SUM(amount), 2) AS total
      FROM transactions WHERE direction = 'out' AND strftime('%Y-%m', occurred_at, ${RIYADH}) = ?
      GROUP BY day ORDER BY day`),
    getSetting: db.prepare('SELECT value FROM settings WHERE key = ?'),
    setSetting: db.prepare(`INSERT INTO settings (key, value) VALUES (?, ?)
      ON CONFLICT(key) DO UPDATE SET value = excluded.value`),
  };

  return {
    close: () => db.close(),

    addTransaction(tx, source) {
      if (tx.raw_text) {
        const since = new Date(Date.now() - DUPLICATE_WINDOW_MS).toISOString();
        const dup = q.recentDuplicate.get(tx.raw_text, since);
        if (dup) return { duplicate: true, transaction: q.txById.get(dup.id) };
      }
      const info = q.insertTx.run(tx.amount, tx.direction, tx.type, tx.merchant || null,
        tx.category, tx.card_last4 || null, tx.note || null, source, tx.raw_text || null,
        tx.occurred_at, nowIso());
      return { duplicate: false, transaction: q.txById.get(Number(info.lastInsertRowid)) };
    },
    transactions: (month) => q.txByMonth.all(month),
    transactionsSince: (iso) => q.txSince.all(iso),
    txCount: () => q.txCount.get().n,
    updateTransaction(id, fields) {
      const cur = q.txById.get(id);
      if (!cur) return null;
      q.updateTx.run(fields.category ?? cur.category, fields.merchant ?? cur.merchant,
        fields.note ?? cur.note, id);
      return q.txById.get(id);
    },
    deleteTransaction: (id) => q.deleteTx.run(id).changes > 0,

    summary(month) {
      const totals = { in: 0, out: 0, count: 0 };
      for (const r of q.totals.all(month)) {
        totals[r.direction] = r.total;
        totals.count += r.n;
      }
      return {
        month,
        spent: totals.out,
        income: totals.in,
        net: Math.round((totals.in - totals.out) * 100) / 100,
        count: totals.count,
        byCategory: q.byCategory.all(month),
        byDay: q.byDay.all(month),
      };
    },

    getSettings() {
      const row = q.getSetting.get('advisor');
      return row ? JSON.parse(row.value) : {};
    },
    saveSettings(settings) {
      q.setSetting.run('advisor', JSON.stringify(settings));
      return settings;
    },
  };
}

// Opens each user's own database on first use and keeps the handle.
// dir === null keeps everything in memory (tests).
function createUserStore(dir) {
  const open = new Map();
  if (dir) fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const fileFor = (userId) => path.join(dir, `u${Number(userId)}.db`);

  return {
    get(userId) {
      const id = Number(userId);
      if (!Number.isSafeInteger(id) || id <= 0) throw new Error('bad user id');
      let store = open.get(id);
      if (!store) {
        store = openUserDb(dir ? fileFor(id) : ':memory:');
        open.set(id, store);
      }
      return store;
    },
    // Closes the user's database and deletes its files.
    remove(userId) {
      const id = Number(userId);
      const store = open.get(id);
      if (store) {
        store.close();
        open.delete(id);
      }
      if (!dir) return;
      for (const suffix of ['', '-wal', '-shm']) {
        try {
          fs.unlinkSync(fileFor(id) + suffix);
        } catch (err) {
          if (err.code !== 'ENOENT') throw err;
        }
      }
    },
    fileFor: (userId) => (dir ? fileFor(userId) : null),
    closeAll() {
      for (const s of open.values()) s.close();
      open.clear();
    },
  };
}

module.exports = { openAuthDb, createUserStore };
