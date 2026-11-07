'use strict';

// Fixtures are synthetic: a made-up salary and made-up merchants.
const test = require('node:test');
const assert = require('node:assert/strict');
const { buildAdvice, cycleFor } = require('../src/advisor');

// 2026-11-07 15:00 Riyadh. With payday 27 the current cycle is Oct 27 → Nov 27.
const NOW = new Date('2026-11-07T12:00:00Z');

// Riyadh local date/time → ISO UTC.
const at = (y, m, d, h = 12) => new Date(Date.UTC(y, m - 1, d, h - 3)).toISOString();
let nextId = 1;
const tx = (fields) => ({ id: nextId++, direction: 'out', type: 'purchase', merchant: null, ...fields });

// Three past cycles of a steady pattern: salary on the 27th, a fixed phone
// bill and subscription, groceries and restaurants that vary.
function history() {
  const out = [];
  const cycles = [[7, 8], [8, 9], [9, 10]]; // cycle start month → next month
  const groceries = [[460, 330, 480, 330], [350, 450, 360, 440], [410, 380, 400, 410]];
  const food = [[100, 180, 95, 160, 230, 135], [160, 140, 130, 220, 155, 95], [130, 200, 100, 150, 190, 130]];
  const stores = ['PANDA', 'DANUBE', 'TAMIMI', 'OTHAIM'];
  const places = ['KUDU', 'ALBAIK', 'BARNS', 'HALF MILLION', 'SHAWARMER', 'DR CAFE'];
  cycles.forEach(([m, next], i) => {
    out.push(tx({ direction: 'in', type: 'salary', amount: 10000, category: 'دخل', occurred_at: at(2026, m, 27, 9) }));
    out.push(tx({ type: 'bill', amount: 300, merchant: 'STC', category: 'فواتير واتصالات', occurred_at: at(2026, next, 5) }));
    out.push(tx({ amount: 56, merchant: 'NETFLIX.COM', category: 'اشتراكات وترفيه', occurred_at: at(2026, next, 10) }));
    groceries[i].forEach((amount, k) => out.push(tx({ amount, merchant: stores[k], category: 'بقالة وتموين', occurred_at: at(2026, next, 1 + k * 6) })));
    food[i].forEach((amount, k) => out.push(tx({ amount, merchant: places[k], category: 'مطاعم وقهوة', occurred_at: at(2026, next, 2 + k * 4) })));
  });
  // The current cycle's salary.
  out.push(tx({ direction: 'in', type: 'salary', amount: 10000, category: 'دخل', occurred_at: at(2026, 10, 27, 9) }));
  return out;
}

test('cycle runs from payday to payday in Riyadh time', () => {
  const c = cycleFor(NOW, 27);
  assert.equal(c.start, '2026-10-26T21:00:00.000Z');
  assert.equal(c.end, '2026-11-26T21:00:00.000Z');
  assert.equal(c.daysTotal, 31);
  assert.equal(c.daysElapsed, 12);
  assert.equal(c.daysLeft, 20);

  // On payday morning a new cycle has started.
  assert.equal(cycleFor(new Date(at(2026, 11, 27, 8)), 27).start, '2026-11-26T21:00:00.000Z');
  // Payday 31 falls on the last day of a short month.
  const feb = cycleFor(new Date(at(2027, 2, 28, 15)), 31);
  assert.equal(feb.start, '2027-02-27T21:00:00.000Z');
  assert.equal(feb.end, '2027-03-30T21:00:00.000Z');
  // Across the year boundary.
  assert.equal(cycleFor(new Date(at(2027, 1, 5)), 27).start, '2026-12-26T21:00:00.000Z');
});

test('no salary: no plan, and the first advice asks for it', () => {
  const a = buildAdvice({ transactions: [], settings: {}, now: NOW });
  assert.equal(a.plan, null);
  assert.equal(a.salary.amount, null);
  assert.equal(a.payday.source, 'default');
  assert.equal(a.insights[0].level, 'alert');
  assert.match(a.insights[0].text, /راتبك/);
});

test('new user with a salary gets the 50/30/20 starting split', () => {
  const a = buildAdvice({ transactions: [], settings: { salary: 10000, payday: 27 }, now: NOW });
  assert.equal(a.plan.basis, 'default');
  assert.equal(a.plan.savings.amount, 2000);
  assert.equal(a.plan.needs, 5000);
  assert.equal(a.plan.wants, 3000);
  assert.equal(a.plan.buffer, 0);
  assert.ok(a.insights.some((i) => /تقسيم عام/.test(i.text)));
});

test('learns salary, payday, averages and fixed commitments from history', () => {
  const a = buildAdvice({ transactions: history(), settings: {}, now: NOW });
  assert.deepEqual(a.salary, { amount: 10000, source: 'detected' });
  assert.deepEqual(a.payday, { day: 27, source: 'detected' });
  assert.equal(a.profile.cyclesAnalyzed, 3);

  const avg = Object.fromEntries(a.profile.avgByCategory.map((c) => [c.category, c.avg]));
  assert.equal(avg['فواتير واتصالات'], 300);
  assert.equal(avg['اشتراكات وترفيه'], 56);
  assert.equal(avg['بقالة وتموين'], 1600);
  assert.equal(avg['مطاعم وقهوة'], 900);

  // Bills and subscriptions are recurring; groceries and restaurants vary, so they are not.
  assert.deepEqual(a.profile.recurring.map((r) => r.merchant), ['STC', 'NETFLIX.COM']);
  assert.equal(a.profile.recurringTotal, 356);

  const budget = Object.fromEntries(a.plan.categories.map((c) => [c.category, c.budget]));
  assert.equal(a.plan.basis, 'history');
  assert.equal(budget['بقالة وتموين'], 1600);
  assert.equal(budget['مطاعم وقهوة'], 900);
  assert.equal(budget['فواتير واتصالات'], 300);
  assert.equal(budget['اشتراكات وترفيه'], 60);
  assert.equal(a.plan.savings.amount, 2000);
  assert.equal(a.plan.buffer, 10000 - 2000 - 1600 - 900 - 300 - 60);
  assert.ok(a.insights.some((i) => i.level === 'good' && /ادخار/.test(i.text)));
  assert.ok(a.insights.some((i) => /التزامات متكررة/.test(i.text)));
});

test('when usual spending exceeds the salary, wants are cut first and fixed bills are kept', () => {
  const a = buildAdvice({ transactions: history(), settings: { salary: 3000, payday: 27 }, now: NOW });
  const budget = Object.fromEntries(a.plan.categories.map((c) => [c.category, c.budget]));
  const total = a.plan.categories.reduce((s, c) => s + c.budget, 0);
  assert.ok(total <= 3000 - a.plan.savings.amount, `budget ${total} must fit`);
  assert.equal(budget['فواتير واتصالات'], 300);
  assert.ok(budget['اشتراكات وترفيه'] >= 60);
  assert.ok(budget['مطاعم وقهوة'] < 900, 'restaurants trimmed');
  assert.ok(budget['بقالة وتموين'] >= 1600 * 0.9 - 10, 'groceries cut by at most ~10%');
  assert.ok(a.plan.buffer >= 0);
});

test('a budget the user sets by hand is kept as is', () => {
  const a = buildAdvice({
    transactions: history(),
    settings: { salary: 3000, payday: 27, budget_overrides: { 'مطاعم وقهوة': 500 } },
    now: NOW,
  });
  const row = a.plan.categories.find((c) => c.category === 'مطاعم وقهوة');
  assert.equal(row.budget, 500);
  assert.equal(row.override, true);
});

test('impossible budgets reduce savings and say so', () => {
  const a = buildAdvice({ transactions: history(), settings: { salary: 1500, payday: 27, savings_pct: 20 }, now: NOW });
  assert.ok(a.plan.savings.amount < 300);
  assert.ok(a.plan.warnings.length > 0);
  assert.equal(a.insights[0].level, 'alert');
});

test('tracks this cycle: overspend, pace, and safe daily amount', () => {
  const txs = history();
  // 12 days into the cycle: 700 on restaurants (budget 900, pace ≈ 1808).
  [200, 150, 180, 170].forEach((amount, k) => txs.push(tx({ amount, merchant: 'KUDU', category: 'مطاعم وقهوة', occurred_at: at(2026, 10, 28 + k) })));
  // 1700 on groceries already (budget 1600): over.
  txs.push(tx({ amount: 1700, merchant: 'PANDA', category: 'بقالة وتموين', occurred_at: at(2026, 11, 2) }));

  const a = buildAdvice({ transactions: txs, settings: {}, now: NOW });
  const by = Object.fromEntries(a.progress.categories.map((c) => [c.category, c]));
  assert.equal(by['بقالة وتموين'].status, 'over');
  assert.equal(by['مطاعم وقهوة'].status, 'warn');
  assert.equal(by['مطاعم وقهوة'].projected, Math.round((700 / 12) * 31 * 100) / 100);
  assert.equal(a.progress.spent, 2400);
  assert.equal(a.progress.income, 10000);
  assert.equal(a.progress.remaining, a.progress.budgetTotal - 2400);
  // STC (300) and Netflix (56) are still due this cycle, so they are set aside.
  assert.equal(a.progress.upcomingFixed, 356);
  assert.equal(a.progress.safeDaily, Math.round(((a.progress.remaining - 356) / 20) * 100) / 100);

  const texts = a.insights.map((i) => i.text).join('\n');
  assert.match(texts, /تجاوزت ميزانية بقالة وتموين بـ 100 ر\.س/);
  assert.match(texts, /بهذا المعدل بتصرف على مطاعم وقهوة/);
  assert.match(texts, /باليوم لين الراتب الجاي بعد ما تحجز 356 ر\.س للالتزامات الباقية \(باقي 20 يوم\)/);
  assert.equal(a.insights[0].level, 'alert');
});

test('settings override detected salary and payday', () => {
  const a = buildAdvice({ transactions: history(), settings: { salary: 12000, payday: 1, savings_pct: 30 }, now: NOW });
  assert.deepEqual(a.salary, { amount: 12000, source: 'settings' });
  assert.deepEqual(a.payday, { day: 1, source: 'settings' });
  assert.equal(a.cycle.start, '2026-10-31T21:00:00.000Z');
  assert.equal(a.plan.savings.amount, 3600);
});

test('a bill paid once is not projected as daily spending', () => {
  const txs = history();
  // STC paid on Nov 5 as usual; Netflix (Nov 10) not yet.
  txs.push(tx({ type: 'bill', amount: 300, merchant: 'STC', category: 'فواتير واتصالات', occurred_at: at(2026, 11, 5) }));
  const a = buildAdvice({ transactions: txs, settings: {}, now: NOW });
  const bills = a.progress.categories.find((c) => c.category === 'فواتير واتصالات');
  assert.equal(bills.projected, 300);
  assert.equal(bills.status, 'ok');
  const subs = a.progress.categories.find((c) => c.category === 'اشتراكات وترفيه');
  assert.equal(subs.projected, 56, 'unpaid subscription is forecast once');
  assert.deepEqual(a.progress.upcoming.map((u) => [u.merchant, u.amount]), [['NETFLIX.COM', 56]]);
  const texts = a.insights.map((i) => i.text).join('\n');
  assert.doesNotMatch(texts, /فواتير واتصالات/, 'no warning about a normal bill');
});

test('"on track" praise is not shown while a budget is over', () => {
  const txs = history();
  txs.push(tx({ amount: 1700, merchant: 'PANDA', category: 'بقالة وتموين', occurred_at: at(2026, 11, 2) }));
  const a = buildAdvice({ transactions: txs, settings: {}, now: NOW });
  assert.ok(!a.insights.some((i) => /أقل من الخطة/.test(i.text)));
});

test('an unnamed bill payment matching a due commitment is counted once', () => {
  const txs = history();
  txs.push(tx({ type: 'bill', amount: 300, merchant: null, category: 'فواتير واتصالات', occurred_at: at(2026, 11, 5) }));
  const a = buildAdvice({ transactions: txs, settings: {}, now: NOW });
  const bills = a.progress.categories.find((c) => c.category === 'فواتير واتصالات');
  assert.equal(bills.projected, 300);
  assert.equal(bills.status, 'ok');
  assert.equal(a.progress.upcomingFixed, 56);
});
