'use strict';

// Sample messages are synthetic: shaped like common Saudi bank alerts,
// with made-up merchants, cards and amounts.
const test = require('node:test');
const assert = require('node:assert/strict');
const { parseSms } = require('../src/parser');

const now = new Date('2026-11-07T12:00:00Z');
const parse = (t) => parseSms(t, { now });

test('Arabic mada purchase with balance after the amount', () => {
  const r = parse('شراء عبر نقاط البيع مدى\nبطاقة: 1234*\nمبلغ: 45.50 ر.س\nلدى: STARBUCKS RIYADH\nفي: 2026-11-05 14:30\nالرصيد: 900.00 ر.س');
  assert.equal(r.ok, true);
  const t = r.transaction;
  assert.equal(t.amount, 45.5);
  assert.equal(t.direction, 'out');
  assert.equal(t.type, 'purchase');
  assert.equal(t.merchant, 'STARBUCKS RIYADH');
  assert.equal(t.category, 'مطاعم وقهوة');
  assert.equal(t.card_last4, '1234');
  assert.equal(t.occurred_at, '2026-11-05T11:30:00.000Z');
});

test('Arabic-Indic digits and thousands separator', () => {
  const r = parse('شراء إنترنت بمبلغ ١٬٢٥٠٫٧٥ ريال من AMAZON.SA بطاقة **٥٥٦٧');
  assert.equal(r.ok, true);
  assert.equal(r.transaction.amount, 1250.75);
  assert.equal(r.transaction.card_last4, '5567');
  assert.equal(r.transaction.category, 'تسوق');
});

test('English purchase, currency before amount', () => {
  const r = parse('Purchase SAR 120.00 at PANDA HYPER on 06/11/2026 21:05 Card ***9876 Avl Bal SAR 3,400.00');
  assert.equal(r.ok, true);
  assert.equal(r.transaction.amount, 120);
  assert.equal(r.transaction.merchant, 'PANDA HYPER');
  assert.equal(r.transaction.category, 'بقالة وتموين');
  assert.equal(r.transaction.card_last4, '9876');
  assert.equal(r.transaction.occurred_at, '2026-11-06T18:05:00.000Z');
});

test('Incoming transfer is income', () => {
  const r = parse('حوالة واردة\nمبلغ: 2000 SAR\nمن: احمد');
  assert.equal(r.ok, true);
  assert.equal(r.transaction.direction, 'in');
  assert.equal(r.transaction.category, 'دخل');
});

test('Outgoing transfer, ATM and bill types', () => {
  assert.equal(parse('حوالة صادرة مبلغ 300 ر.س').transaction.type, 'transfer');
  assert.equal(parse('سحب نقدي من صراف آلي مبلغ 500 SAR').transaction.type, 'withdrawal');
  const bill = parse('سداد فاتورة STC بمبلغ 230 ريال');
  assert.equal(bill.transaction.type, 'bill');
  assert.equal(bill.transaction.category, 'فواتير واتصالات');
});

test('Refund counts as money in', () => {
  const r = parse('Refund SAR 60.00 from NOON');
  assert.equal(r.transaction.direction, 'in');
  assert.equal(r.transaction.type, 'refund');
});

test('Message without amount is rejected, not guessed', () => {
  assert.equal(parse('رمز التحقق الخاص بك هو 4821').ok, false);
  assert.equal(parse('').ok, false);
  assert.equal(parseSms(null).ok, false);
});

test('Missing date falls back to now', () => {
  assert.equal(parse('شراء 15 ر.س لدى كافيه').transaction.occurred_at, now.toISOString());
});

test('Invalid date falls back to now', () => {
  assert.equal(parse('شراء 15 ر.س بتاريخ 45/13/2026').transaction.occurred_at, now.toISOString());
});
