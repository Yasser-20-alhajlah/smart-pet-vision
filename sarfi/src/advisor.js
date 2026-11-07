'use strict';

// Personal budgeting advisor. Pure functions: given one user's transactions
// and settings, it learns their spending pattern from past salary cycles and
// proposes how to split this cycle's salary, then tracks progress against it.
// Rule-based and local: no data leaves the server. General budgeting
// guidance, not licensed financial advice.

const DAY = 86400000;
const RIYADH_OFFSET = 3 * 3600 * 1000; // UTC+3, no DST
const HISTORY_CYCLES = 3;
const DEFAULT_PAYDAY = 27;
const DEFAULT_SAVINGS_PCT = 20;

const GROUPS = {
  needs: ['بقالة وتموين', 'فواتير واتصالات', 'مواصلات ووقود', 'صحة'],
  wants: ['مطاعم وقهوة', 'تسوق', 'اشتراكات وترفيه', 'سفر', 'أخرى'],
};
const GROUP_OF = Object.fromEntries(Object.entries(GROUPS).flatMap(([g, cs]) => cs.map((c) => [c, g])));
const SPEND_CATEGORIES = [...GROUPS.needs, ...GROUPS.wants];

// Starting split (share of salary) for a user with no history yet:
// needs 50%, wants 30%, savings 20%.
const DEFAULT_SHARES = {
  'بقالة وتموين': 0.20, 'فواتير واتصالات': 0.12, 'مواصلات ووقود': 0.12, 'صحة': 0.06,
  'مطاعم وقهوة': 0.10, 'تسوق': 0.08, 'اشتراكات وترفيه': 0.03, 'سفر': 0.04, 'أخرى': 0.05,
};

const WEEKDAYS = ['الأحد', 'الاثنين', 'الثلاثاء', 'الأربعاء', 'الخميس', 'الجمعة', 'السبت'];

const round2 = (n) => Math.round(n * 100) / 100;
const round10 = (n) => Math.round(n / 10) * 10;
const ceil10 = (n) => Math.ceil(n / 10) * 10;
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const sar = (n) => `${Math.round(n).toLocaleString('en-US')} ر.س`;

function median(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function mode(xs) {
  const counts = new Map();
  for (const x of xs) counts.set(x, (counts.get(x) || 0) + 1);
  let best = null;
  let bestN = 0;
  for (const [x, n] of counts) if (n > bestN || (n === bestN && x > best)) [best, bestN] = [x, n];
  return best;
}

const riyadh = (date) => new Date(new Date(date).getTime() + RIYADH_OFFSET);
const daysInMonth = (y, m) => new Date(Date.UTC(y, m + 1, 0)).getUTCDate();
const riyadhMidnight = (y, m, d) => new Date(Date.UTC(y, m, d) - RIYADH_OFFSET);

// The salary cycle containing `now`: from payday (Riyadh midnight) to the next
// payday. A payday past the end of a short month falls on its last day.
function cycleFor(now, payday) {
  const r = riyadh(now);
  const y = r.getUTCFullYear();
  const m = r.getUTCMonth();
  const pd = (yy, mm) => Math.min(payday, daysInMonth(yy, mm));
  let sy = y;
  let sm = m;
  if (r.getUTCDate() < pd(y, m)) {
    sm -= 1;
    if (sm < 0) { sm = 11; sy -= 1; }
  }
  let ey = sy;
  let em = sm + 1;
  if (em > 11) { em = 0; ey += 1; }
  const start = riyadhMidnight(sy, sm, pd(sy, sm));
  const end = riyadhMidnight(ey, em, pd(ey, em));
  const daysTotal = Math.round((end - start) / DAY);
  const daysElapsed = Math.min(daysTotal, Math.floor((new Date(now) - start) / DAY) + 1);
  return {
    start: start.toISOString(),
    end: end.toISOString(),
    daysTotal,
    daysElapsed,
    daysLeft: daysTotal - daysElapsed + 1, // including today
  };
}

function previousCycles(current, payday, n) {
  const out = [];
  let c = current;
  for (let i = 0; i < n; i += 1) {
    c = cycleFor(new Date(new Date(c.start).getTime() - 1), payday);
    out.push(c);
  }
  return out;
}

const inCycle = (t, c) => t.occurred_at >= c.start && t.occurred_at < c.end;

function normalizeMerchant(m) {
  if (!m) return null;
  const key = m.toLowerCase().replace(/[\d*#.,:;\-_/\\]+/g, ' ').replace(/\s+/g, ' ').trim();
  return key || null;
}

// Salary and payday learned from salary SMS, unless the user set them.
function detectIncome(transactions, settings) {
  const salaries = transactions.filter((t) => t.direction === 'in' && t.type === 'salary');
  const detectedAmount = median(salaries.slice(-3).map((t) => t.amount));
  const detectedDay = mode(salaries.map((t) => riyadh(t.occurred_at).getUTCDate()));

  let salary = { amount: null, source: null };
  if (settings.salary > 0) salary = { amount: settings.salary, source: 'settings' };
  else if (detectedAmount) salary = { amount: round2(detectedAmount), source: 'detected' };

  let payday = { day: DEFAULT_PAYDAY, source: 'default' };
  if (settings.payday >= 1) payday = { day: settings.payday, source: 'settings' };
  else if (detectedDay) payday = { day: detectedDay, source: 'detected' };
  return { salary, payday };
}

// How this user spends, learned from up to HISTORY_CYCLES past cycles.
function buildProfile(transactions, history) {
  const spend = transactions.filter((t) => t.direction === 'out');
  const perCycle = history.map((c) => spend.filter((t) => inCycle(t, c)));
  const active = perCycle.filter((txs) => txs.length > 0);
  const n = active.length;

  const avgByCategory = {};
  for (const cat of SPEND_CATEGORIES) {
    const total = sum(active.map((txs) => sum(txs.filter((t) => t.category === cat).map((t) => t.amount))));
    avgByCategory[cat] = n ? round2(total / n) : 0;
  }

  // Recurring (bills, subscriptions, rent): same merchant in 2+ cycles, at
  // most ~2 payments per cycle, every payment within 10% of the median.
  const byMerchant = new Map();
  perCycle.forEach((txs, i) => {
    for (const t of txs) {
      const key = normalizeMerchant(t.merchant);
      if (!key) continue;
      if (!byMerchant.has(key)) byMerchant.set(key, { name: t.merchant, cycles: new Map(), cats: [] });
      const entry = byMerchant.get(key);
      entry.cycles.set(i, [...(entry.cycles.get(i) || []), t.amount]);
      entry.cats.push(t.category);
    }
  });
  const recurring = [];
  for (const entry of byMerchant.values()) {
    if (entry.cycles.size < 2) continue;
    const totals = [...entry.cycles.values()].map(sum);
    const payments = [...entry.cycles.values()].flat();
    if (payments.length / entry.cycles.size > 2) continue;
    const mid = median(payments);
    if (payments.some((a) => Math.abs(a - mid) > mid * 0.1)) continue;
    recurring.push({ merchant: entry.name, category: mode(entry.cats), monthly: round2(median(totals)) });
  }
  recurring.sort((a, b) => b.monthly - a.monthly);

  const windowSpend = active.flat();
  const merchantTotals = new Map();
  for (const t of windowSpend) {
    const key = normalizeMerchant(t.merchant);
    if (!key) continue;
    const e = merchantTotals.get(key) || { merchant: t.merchant, total: 0, count: 0 };
    e.total += t.amount;
    e.count += 1;
    merchantTotals.set(key, e);
  }
  const topMerchants = [...merchantTotals.values()]
    .sort((a, b) => b.total - a.total).slice(0, 5)
    .map((e) => ({ ...e, total: round2(e.total) }));

  const weekday = new Array(7).fill(0);
  for (const t of windowSpend) weekday[riyadh(t.occurred_at).getUTCDay()] += t.amount;
  const windowTotal = sum(weekday);
  const peak = weekday.indexOf(Math.max(...weekday));

  return {
    cyclesAnalyzed: n,
    avgMonthlySpend: n ? round2(sum(windowSpend.map((t) => t.amount)) / n) : 0,
    avgByCategory: Object.entries(avgByCategory)
      .filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1])
      .map(([category, avg]) => ({ category, avg })),
    recurring,
    recurringTotal: round2(sum(recurring.map((r) => r.monthly))),
    topMerchants,
    peakWeekday: windowTotal ? { day: WEEKDAYS[peak], share: round2(weekday[peak] / windowTotal) } : null,
    weekendShare: windowTotal ? round2((weekday[5] + weekday[6]) / windowTotal) : 0,
    avg: avgByCategory,
  };
}

// Splits the salary: savings first, then fixed commitments, then the rest
// from the user's own averages. If that doesn't fit, wants are trimmed
// before needs, and savings are reduced only as a last resort.
function buildPlan(salary, settings, profile) {
  if (!salary) return null;
  const warnings = [];
  const savingsPct = Number.isFinite(settings.savings_pct) ? settings.savings_pct : DEFAULT_SAVINGS_PCT;
  let savings = round10((salary * savingsPct) / 100);
  const overrides = settings.budget_overrides || {};
  const hasHistory = profile.cyclesAnalyzed > 0;

  const fixedBy = {};
  for (const r of profile.recurring) {
    if (GROUP_OF[r.category]) fixedBy[r.category] = (fixedBy[r.category] || 0) + r.monthly;
  }

  const rows = SPEND_CATEGORIES.map((category) => {
    const fixed = ceil10(fixedBy[category] || 0);
    const baseline = hasHistory ? profile.avg[category] : salary * DEFAULT_SHARES[category];
    const override = Number.isFinite(overrides[category]) ? overrides[category] : null;
    const budget = override ?? Math.max(fixed, ceil10(baseline));
    return { category, group: GROUP_OF[category], baseline: round2(baseline), fixed, budget, override: override !== null };
  });

  const total = () => sum(rows.map((r) => r.budget));
  let excess = total() - (salary - savings);

  // Trim to fit, wants first. Overridden budgets and fixed commitments are kept.
  const trim = (group, maxShare) => {
    if (excess <= 0) return;
    const cuttable = rows.filter((r) => r.group === group && !r.override)
      .map((r) => ({ r, room: Math.max(0, (r.budget - r.fixed) * maxShare) }));
    const room = sum(cuttable.map((c) => c.room));
    if (room <= 0) return;
    const take = Math.min(excess, room);
    for (const c of cuttable) {
      c.r.budget = Math.max(c.r.fixed, Math.floor((c.r.budget - (take * c.room) / room) / 10) * 10);
    }
    excess = total() - (salary - savings);
  };
  trim('wants', 0.5);
  trim('needs', 0.1);
  trim('wants', 1);

  if (excess > 0) {
    const cut = Math.min(savings, ceil10(excess));
    if (cut > 0) {
      savings -= cut;
      warnings.push(`صرفك المعتاد أكبر من راتبك بعد الادخار، فخفضت الادخار إلى ${sar(savings)}.`);
    }
    excess = total() - (salary - savings);
    if (excess > 0) warnings.push(`الميزانية أكبر من راتبك بـ ${sar(excess)}. لازم تخفض بعض البنود.`);
  }

  const groupTotal = (g) => sum(rows.filter((r) => r.group === g).map((r) => r.budget));
  return {
    salary,
    basis: hasHistory ? 'history' : 'default',
    savings: { pct: round2((savings / salary) * 100), amount: savings, targetPct: savingsPct },
    needs: groupTotal('needs'),
    wants: groupTotal('wants'),
    fixedTotal: sum(rows.map((r) => r.fixed)),
    buffer: round2(salary - savings - total()),
    categories: rows.filter((r) => r.budget > 0),
    warnings,
  };
}

// Progress through the current cycle. Recurring payments (rent, bills,
// subscriptions) are counted once, not extrapolated: the forecast is what
// was paid + recurring payments still due + variable spending at its pace.
function buildProgress(transactions, cycle, plan, profile) {
  const txs = transactions.filter((t) => inCycle(t, cycle));
  const out = txs.filter((t) => t.direction === 'out');
  const income = round2(sum(txs.filter((t) => t.direction === 'in').map((t) => t.amount)));
  const canProject = cycle.daysElapsed >= 3;

  const recurringByKey = new Map(profile.recurring.map((r) => [normalizeMerchant(r.merchant), { ...r, paid: 0 }]));
  const by = {};
  const slot = (c) => (by[c] ||= { spent: 0, recurringPaid: 0, upcoming: 0 });
  // A payment with no merchant name (common in bill SMS) counts as a recurring
  // payment still due in the same category when the amount is within 10%.
  const matchUnnamed = (t) => [...recurringByKey.values()].find((r) => r.category === t.category
    && r.paid === 0 && Math.abs(t.amount - r.monthly) <= r.monthly * 0.1);
  for (const t of out) {
    const s = slot(t.category);
    s.spent += t.amount;
    const r = t.merchant ? recurringByKey.get(normalizeMerchant(t.merchant)) : matchUnnamed(t);
    if (r) {
      s.recurringPaid += t.amount;
      r.paid += t.amount;
    }
  }
  const upcoming = [];
  for (const r of recurringByKey.values()) {
    const due = round2(Math.max(0, r.monthly - r.paid));
    if (due > 0) {
      slot(r.category).upcoming += due;
      upcoming.push({ merchant: r.merchant, category: r.category, amount: due });
    }
  }
  const forecast = (s) => {
    if (!canProject) return null;
    const variable = s.spent - s.recurringPaid;
    return round2(s.recurringPaid + s.upcoming + (variable / cycle.daysElapsed) * cycle.daysTotal);
  };

  const categories = SPEND_CATEGORIES
    .filter((c) => (plan && plan.categories.some((r) => r.category === c)) || by[c]?.spent)
    .map((category) => {
      const row = plan?.categories.find((r) => r.category === category);
      const budget = row?.budget ?? 0;
      const s = by[category] || { spent: 0, recurringPaid: 0, upcoming: 0 };
      const spent = round2(s.spent);
      const projected = forecast(s);
      const variableBudget = budget - (row?.fixed ?? 0);
      const variableSpent = s.spent - s.recurringPaid;
      let status = 'ok';
      if (spent > budget) status = 'over';
      else if (budget && ((projected !== null && projected > budget * 1.1)
        || (variableBudget > 0 && variableSpent >= variableBudget * 0.8))) status = 'warn';
      return { category, budget, spent, pct: budget ? round2(spent / budget) : null, projected, status };
    })
    .sort((a, b) => b.spent - a.spent);

  const spent = round2(sum(out.map((t) => t.amount)));
  const upcomingFixed = round2(sum(upcoming.map((u) => u.amount)));
  const budgetTotal = plan ? sum(plan.categories.map((r) => r.budget)) + Math.max(0, plan.buffer) : null;
  const remaining = budgetTotal === null ? null : round2(budgetTotal - spent);
  const projected = canProject ? round2(sum(Object.values(by).map(forecast))) : null;
  return {
    spent,
    income,
    projected,
    budgetTotal,
    remaining,
    upcomingFixed,
    upcoming,
    // Free to spend per day after setting aside recurring payments still due.
    safeDaily: remaining === null ? null : round2(Math.max(0, remaining - upcomingFixed) / cycle.daysLeft),
    categories,
  };
}

const LEVEL_ORDER = { alert: 0, warn: 1, good: 2, info: 3 };

function buildInsights({ salary, payday, plan, progress, profile, cycle }) {
  const out = [];
  const add = (level, text) => out.push({ level, text });

  if (!salary.amount) {
    add('alert', 'حدد راتبك عشان أوزّعه لك، أو أضف رسالة إيداع الراتب وأكتشفه تلقائياً.');
  } else if (salary.source === 'detected') {
    add('info', `اكتشفت راتبك تلقائياً: ${sar(salary.amount)} يوم ${payday.day} من كل شهر. عدّله من الأسفل إذا مو صحيح.`);
  }

  if (plan) {
    for (const w of plan.warnings) add('alert', w);
    for (const c of progress.categories) {
      if (c.status === 'over') {
        add('alert', `تجاوزت ميزانية ${c.category} بـ ${sar(c.spent - c.budget)}.`);
      } else if (c.status === 'warn' && c.projected !== null && c.projected > c.budget) {
        add('warn', `بهذا المعدل بتصرف على ${c.category} حوالي ${sar(c.projected)}، والميزانية ${sar(c.budget)}.`);
      } else if (c.status === 'warn') {
        add('warn', `صرفت ${Math.round(c.pct * 100)}٪ من ميزانية ${c.category}.`);
      }
    }
    if (progress.remaining > 0) {
      const aside = progress.upcomingFixed > 0 ? ` بعد ما تحجز ${sar(progress.upcomingFixed)} للالتزامات الباقية` : '';
      add('info', `تقدر تصرف ${sar(progress.safeDaily)} باليوم لين الراتب الجاي${aside} (باقي ${cycle.daysLeft} يوم).`);
    } else if (progress.remaining !== null) {
      add('alert', `خلصت ميزانية هذا الشهر وباقي ${cycle.daysLeft} يوم. وقّف الصرف غير الضروري.`);
    }
    if (plan.buffer >= 100) {
      add('good', `عندك ${sar(plan.buffer)} زيادة عن صرفك المعتاد. لو حولتها للادخار يصير ادخارك ${sar(plan.savings.amount + plan.buffer)}.`);
    }
    if (plan.savings.pct < 10) {
      add('warn', `ادخارك ${Math.round(plan.savings.pct)}٪ من الراتب. حاول توصله على الأقل 10٪.`);
    }
    const onTrack = progress.categories.every((c) => c.status === 'ok');
    if (onTrack && progress.projected !== null && progress.budgetTotal && progress.projected <= progress.budgetTotal * 0.9) {
      add('good', 'صرفك هذا الشهر ماشي أقل من الخطة. استمر.');
    }
  }

  if (profile.cyclesAnalyzed > 0) {
    if (profile.recurringTotal > 0) {
      const names = profile.recurring.slice(0, 3).map((r) => r.merchant).join('، ');
      add('info', `عندك التزامات متكررة بحوالي ${sar(profile.recurringTotal)} شهرياً (${names}). راجعها وألغِ اللي ما تحتاجه.`);
    }
    for (const c of progress.categories) {
      const avg = profile.avg[c.category] || 0;
      if (avg >= 100 && c.projected !== null && c.projected > avg * 1.3) {
        add('warn', `صرفك على ${c.category} هذا الشهر ماشي أعلى من معدلك (${sar(avg)}) بأكثر من 30٪.`);
      }
    }
    if (profile.weekendShare >= 0.45) {
      add('info', `${Math.round(profile.weekendShare * 100)}٪ من صرفك يكون الجمعة والسبت. خطط لنهاية الأسبوع مسبقاً.`);
    }
    if (profile.avgMonthlySpend > 0) {
      add('info', `صندوق الطوارئ المقترح: ${sar(profile.avgMonthlySpend * 3)} إلى ${sar(profile.avgMonthlySpend * 6)} (٣ إلى ٦ أشهر من صرفك).`);
    }
  } else if (salary.amount) {
    add('info', 'الخطة الحين تقسيم عام (50٪ احتياجات، 30٪ رغبات، 20٪ ادخار). بعد دورة راتب كاملة من تسجيل مصاريفك أبنيها على طريقة صرفك أنت.');
  }

  return out.sort((a, b) => LEVEL_ORDER[a.level] - LEVEL_ORDER[b.level]);
}

// Earliest timestamp the advisor reads: enough for HISTORY_CYCLES past
// cycles plus the current one, whatever the payday.
function historyStart(now) {
  return new Date(new Date(now).getTime() - (HISTORY_CYCLES + 2) * 31 * DAY).toISOString();
}

function buildAdvice({ transactions, settings = {}, now = new Date() }) {
  const { salary, payday } = detectIncome(transactions, settings);
  const cycle = cycleFor(now, payday.day);
  const history = previousCycles(cycle, payday.day, HISTORY_CYCLES);
  const profile = buildProfile(transactions, history);
  const plan = buildPlan(salary.amount, settings, profile);
  const progress = buildProgress(transactions, cycle, plan, profile);
  const insights = buildInsights({ salary, payday, plan, progress, profile, cycle });
  const { avg, ...publicProfile } = profile;
  return { cycle, salary, payday, profile: publicProfile, plan, progress, insights };
}

module.exports = { buildAdvice, cycleFor, historyStart, GROUPS, SPEND_CATEGORIES, DEFAULT_SAVINGS_PCT };
