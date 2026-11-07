'use strict';

// Parses Saudi bank SMS / notification text (Arabic or English) into a
// transaction. Bank formats vary and change, so every field is best-effort:
// a message with no recognisable amount is rejected rather than guessed.

const ARABIC_DIGITS = '٠١٢٣٤٥٦٧٨٩';
const PERSIAN_DIGITS = '۰۱۲۳۴۵۶۷۸۹';

function normalizeDigits(text) {
  return text
    .replace(/[٠-٩]/g, (d) => String(ARABIC_DIGITS.indexOf(d)))
    .replace(/[۰-۹]/g, (d) => String(PERSIAN_DIGITS.indexOf(d)))
    .replace(/٫/g, '.')
    .replace(/٬/g, ',');
}

const CURRENCY = '(?:SAR|SR|ر\\.?\\s?س\\.?|ريال(?:\\s?سعودي)?|﷼)';
const NUMBER = '(\\d{1,3}(?:,\\d{3})+(?:\\.\\d{1,2})?|\\d+(?:\\.\\d{1,2})?)';

// Amount is usually written "SAR 45.50", "45.50 SAR", "مبلغ: 45.50 ر.س" or
// "بـ 45.50 ريال". Labelled amounts win over bare currency matches so that a
// trailing "الرصيد: 900 SAR" is not mistaken for the spend.
const AMOUNT_PATTERNS = [
  new RegExp(`(?:مبلغ|المبلغ|بمبلغ|Amount|Amt)\\s*[:：]?\\s*${CURRENCY}?\\s*${NUMBER}`, 'gi'),
  new RegExp(`${CURRENCY}\\s*${NUMBER}`, 'gi'),
  new RegExp(`${NUMBER}\\s*${CURRENCY}`, 'gi'),
];

const BALANCE_RE = /(?:الرصيد|رصيد|Balance|Bal|Avl|Available)\s*[:：]?\s*$/i;

const TYPES = [
  { type: 'refund', re: /استرداد|استرجاع|refund|reversal/i, direction: 'in' },
  { type: 'salary', re: /راتب|salary|payroll/i, direction: 'in' },
  { type: 'incoming', re: /حوالة\s*واردة|إيداع|ايداع|incoming|credited|deposit|received/i, direction: 'in' },
  { type: 'withdrawal', re: /سحب|صراف|ATM|withdraw/i, direction: 'out' },
  { type: 'transfer', re: /حوالة|تحويل|transfer/i, direction: 'out' },
  { type: 'bill', re: /سداد|فاتورة|bill|SADAD/i, direction: 'out' },
  { type: 'purchase', re: /شراء|مشتريات|purchase|POS|Apple\s?Pay|مدى|mada|online|انترنت|إنترنت/i, direction: 'out' },
];

// Merchant follows "لدى/من/عند/at/from". Stop at the next labelled field or line.
const MERCHANT_RE = /(?:لدى|عند|التاجر|من\s*:|\bAt\b|\bat\b|\bfrom\b|Merchant)\s*[:：]?\s*([^\n:،,]+?)(?=\s*(?:\n|$|مبلغ|المبلغ|بمبلغ|في\s+\d|بتاريخ|تاريخ|\bon\s+\d|الرصيد|رصيد|بطاقة|\bCard\b|Amount|SAR|ر\.?س|ريال|\d{1,2}[/-]\d{1,2}))/;

const CARD_RE = /(?:بطاقة|البطاقة|Card|card|حساب|Acc(?:ount)?)[^\d\n]{0,12}?[*xX#]*(\d{4})\b|[*xX]{2,}(\d{4})\b/;

const DATE_RE = /(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:\s+(\d{1,2}):(\d{2}))?|(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})(?:\s+(\d{1,2}):(\d{2}))?/;

const CATEGORIES = [
  ['مطاعم وقهوة', /مطعم|كوفي|كافيه|قهوة|cafe|coffee|starbucks|ستاربكس|kudu|كودو|البيك|albaik|mcdonald|ماكدونالدز|burger|برجر|pizza|بيتزا|hungerstation|هنقرستيشن|jahez|جاهز|marsool|مرسول|toyou|تويو|shawarma|شاورما|dunkin|barn|بارنز|half million|هاف مليون/i],
  ['بقالة وتموين', /بقالة|تموينات|سوبرماركت|هايبر|panda|بنده|danube|الدانوب|othaim|العثيمين|tamimi|التميمي|carrefour|كارفور|lulu|لولو|nana|نعناع/i],
  ['مواصلات ووقود', /uber|اوبر|أوبر|careem|كريم|بنزين|محطة|petrol|fuel|aldrees|الدريس|sasco|ساسكو|jeeny|جيني|bolt/i],
  ['تسوق', /amazon|امازون|أمازون|noon|نون|shein|namshi|نمشي|ikea|ايكيا|extra|اكسترا|jarir|جرير|centrepoint|سنتربوينت|zara|زارا/i],
  ['فواتير واتصالات', /\bstc\b|موبايلي|mobily|\bzain\b|زين|كهرباء|electric|مياه|\bwater\b|سداد|sadad|فاتورة|\bbill\b|اتصالات/i],
  ['اشتراكات وترفيه', /netflix|نتفلكس|shahid|شاهد|spotify|سبوتيفاي|apple\.com|itunes|google|youtube|\bosn\b|anghami|أنغامي|playstation|steam|cinema|سينما|\bvox\b|muvi/i],
  ['صحة', /صيدلية|pharmacy|nahdi|النهدي|الدواء|aldawaa|مستشفى|hospital|عيادة|clinic|مختبر/i],
  ['سفر', /flynas|saudia|flyadeal|طيران|airline|booking\.com|hotel|فندق|airbnb|almosafer|المسافر/i],
];

const CATEGORY_NAMES = [...CATEGORIES.map((c) => c[0]), 'أخرى', 'دخل'];

function parseAmount(text) {
  for (const re of AMOUNT_PATTERNS) {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(text)) !== null) {
      // Skip amounts that sit right after a balance label.
      const before = text.slice(Math.max(0, m.index - 24), m.index);
      if (BALANCE_RE.test(before)) continue;
      const value = Number(m[1].replace(/,/g, ''));
      if (Number.isFinite(value) && value > 0) return Math.round(value * 100) / 100;
    }
  }
  return null;
}

function parseDate(text, now) {
  const m = DATE_RE.exec(text);
  if (!m) return now.toISOString();
  let y;
  let mo;
  let d;
  let h = 0;
  let mi = 0;
  if (m[1]) {
    [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
    if (m[4]) [h, mi] = [Number(m[4]), Number(m[5])];
  } else {
    [d, mo, y] = [Number(m[6]), Number(m[7]), Number(m[8])];
    if (y < 100) y += 2000;
    if (m[9]) [h, mi] = [Number(m[9]), Number(m[10])];
  }
  if (mo < 1 || mo > 12 || d < 1 || d > 31 || y < 2000 || y > 2100 || h > 23 || mi > 59) {
    return now.toISOString();
  }
  // Saudi bank times are local (UTC+3, no DST).
  const date = new Date(Date.UTC(y, mo - 1, d, h - 3, mi));
  return Number.isNaN(date.getTime()) ? now.toISOString() : date.toISOString();
}

function categorize(merchant, text) {
  const haystack = `${merchant || ''} ${text || ''}`;
  for (const [name, re] of CATEGORIES) if (re.test(haystack)) return name;
  return 'أخرى';
}

function parseSms(rawText, { now = new Date() } = {}) {
  if (typeof rawText !== 'string' || !rawText.trim()) {
    return { ok: false, error: 'النص فارغ' };
  }
  const text = normalizeDigits(rawText).replace(/\r/g, '').trim();
  const amount = parseAmount(text);
  if (amount === null) return { ok: false, error: 'ما لقيت مبلغ في الرسالة' };

  const typeInfo = TYPES.find((t) => t.re.test(text)) || { type: 'purchase', direction: 'out' };
  const merchantMatch = MERCHANT_RE.exec(text);
  const merchant = merchantMatch ? merchantMatch[1].trim().slice(0, 80) || null : null;
  const cardMatch = CARD_RE.exec(text);

  return {
    ok: true,
    transaction: {
      amount,
      direction: typeInfo.direction,
      type: typeInfo.type,
      merchant,
      category: typeInfo.direction === 'in' ? 'دخل' : categorize(merchant, text),
      card_last4: cardMatch ? cardMatch[1] || cardMatch[2] : null,
      occurred_at: parseDate(text, now),
      raw_text: rawText.slice(0, 1000),
    },
  };
}

module.exports = { parseSms, normalizeDigits, categorize, CATEGORY_NAMES };
