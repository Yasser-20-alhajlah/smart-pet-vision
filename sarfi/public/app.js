'use strict';

(() => {
  const TOKEN_KEY = 'sarfi.token';
  const $ = (id) => document.getElementById(id);
  const fmt = new Intl.NumberFormat('ar-SA', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const money = (n) => `${fmt.format(n || 0)} ر.س`;
  const dateFmt = new Intl.DateTimeFormat('ar-SA', { timeZone: 'Asia/Riyadh', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
  const monthFmt = new Intl.DateTimeFormat('ar-SA', { timeZone: 'UTC', month: 'long', year: 'numeric', calendar: 'gregory' });

  const state = { token: null, me: null, categories: [], month: riyadhMonth(), tx: [], editing: null };

  function riyadhMonth() {
    return new Date(Date.now() + 3 * 3600 * 1000).toISOString().slice(0, 7);
  }
  function shiftMonth(m, delta) {
    const [y, mo] = m.split('-').map(Number);
    const d = new Date(Date.UTC(y, mo - 1 + delta, 1));
    return d.toISOString().slice(0, 7);
  }

  function storeGet() { try { return localStorage.getItem(TOKEN_KEY); } catch { return null; } }
  function storeSet(v) { try { if (v) localStorage.setItem(TOKEN_KEY, v); else localStorage.removeItem(TOKEN_KEY); } catch { /* private mode */ } }

  let toastTimer;
  function toast(msg) {
    const t = $('toast');
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), 2500);
  }

  async function api(method, path, body) {
    const headers = {};
    if (state.token) headers.Authorization = `Bearer ${state.token}`;
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
    if (res.status === 401 && state.token) {
      logout();
      throw new Error('انتهت الجلسة، ادخل من جديد');
    }
    const data = res.headers.get('content-type')?.includes('json') ? await res.json() : null;
    if (!res.ok) throw new Error(data?.error || 'صار خطأ');
    return data;
  }

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const c of children) if (c != null) node.append(c);
    return node;
  }

  // ---------- views ----------
  function show(view) {
    for (const s of document.querySelectorAll('.view')) s.hidden = s.id !== `view-${view}`;
    for (const b of document.querySelectorAll('.tab')) b.classList.toggle('active', b.dataset.view === view);
    if (view === 'home') loadHome();
    if (view === 'settings') loadSettings();
    if (view === 'owner') loadOwner();
  }

  async function boot() {
    const joinCode = new URLSearchParams(location.hash.slice(1)).get('join');
    state.token = storeGet();
    if (state.token) {
      try {
        const r = await api('GET', '/api/me');
        state.me = r.user;
        state.categories = r.categories;
      } catch {
        state.token = null;
      }
    }
    if (!state.me) {
      $('nav').hidden = true;
      if (joinCode) $('joinForm').code.value = joinCode;
      history.replaceState(null, '', location.pathname);
      show('join');
      return;
    }
    history.replaceState(null, '', location.pathname);
    $('nav').hidden = false;
    $('ownerTab').hidden = state.me.role !== 'owner';
    fillCategorySelects();
    show('home');
  }

  function logout() {
    state.token = null;
    state.me = null;
    storeSet(null);
    $('nav').hidden = true;
    show('join');
  }

  function fillCategorySelects() {
    for (const id of ['manualCategory', 'editCategory']) {
      const sel = $(id);
      sel.replaceChildren(...state.categories.filter((c) => c !== 'دخل').map((c) => el('option', { value: c, text: c })));
    }
  }

  // ---------- home ----------
  async function loadHome() {
    const [y, m] = state.month.split('-').map(Number);
    $('monthLabel').textContent = monthFmt.format(new Date(Date.UTC(y, m - 1, 1)));
    $('nextMonth').disabled = state.month >= riyadhMonth();
    try {
      const q = `?month=${state.month}`;
      const [sum, list] = await Promise.all([api('GET', `/api/summary${q}`), api('GET', `/api/transactions${q}`)]);
      renderSummary(sum);
      state.tx = list.transactions;
      renderTx();
    } catch (e) {
      toast(e.message);
    }
  }

  function renderSummary(sum) {
    $('statSpent').textContent = money(sum.spent);
    $('statIncome').textContent = money(sum.income);
    $('statNet').textContent = money(sum.net);
    const max = Math.max(...sum.byCategory.map((c) => c.total), 1);
    $('categoryList').replaceChildren(...sum.byCategory.map((c) => {
      const fill = el('span');
      fill.style.width = `${Math.max(2, (c.total / max) * 100)}%`;
      return el('li', {},
        el('div', { class: 'bar-head' }, el('span', { text: `${c.category} (${c.n})` }), el('b', { text: money(c.total) })),
        el('div', { class: 'bar' }, fill));
    }));
    $('categoryEmpty').hidden = sum.byCategory.length > 0;
  }

  function renderTx() {
    $('txList').replaceChildren(...state.tx.map((t) => el('li', {},
      el('button', { type: 'button', class: 'tx-item', onclick: () => openEdit(t) },
        el('span', { class: 'tx-main' },
          el('b', { text: t.merchant || t.category }),
          el('small', { text: `${t.category} · ${dateFmt.format(new Date(t.occurred_at))}${t.card_last4 ? ` · ${t.card_last4}` : ''}` })),
        el('span', { class: `tx-amount ${t.direction}`, text: `${t.direction === 'in' ? '+' : '−'}${money(t.amount)}` })))));
    $('txEmpty').hidden = state.tx.length > 0;
  }

  // ---------- dialogs ----------
  function openDialog(id) {
    const d = $(id);
    d.querySelectorAll('.error').forEach((e) => { e.textContent = ''; });
    d.showModal();
  }
  document.addEventListener('click', (e) => {
    const close = e.target.closest('[data-close]');
    if (close) close.closest('dialog').close();
  });

  $('openSms').addEventListener('click', () => {
    $('smsForm').reset();
    openDialog('smsDialog');
  });
  $('smsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      const r = await api('POST', '/api/sms', { text: e.target.text.value });
      $('smsDialog').close();
      toast(r.duplicate ? 'هذي الرسالة مسجلة من قبل' : `تمت إضافة ${money(r.transaction.amount)}`);
      loadHome();
    } catch (err) {
      $('smsError').textContent = err.message;
    }
  });

  $('openManual').addEventListener('click', () => {
    const f = $('manualForm');
    f.reset();
    const local = new Date(Date.now() - new Date().getTimezoneOffset() * 60000);
    f.date.value = local.toISOString().slice(0, 16);
    openDialog('manualDialog');
  });
  $('manualForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = e.target;
    try {
      await api('POST', '/api/transactions', {
        amount: Number(f.amount.value),
        direction: f.direction.value,
        merchant: f.merchant.value,
        category: f.category.value,
        note: f.note.value,
        occurred_at: new Date(f.date.value).toISOString(),
      });
      $('manualDialog').close();
      toast('تمت الإضافة');
      loadHome();
    } catch (err) {
      $('manualError').textContent = err.message;
    }
  });

  function openEdit(t) {
    state.editing = t;
    const f = $('editForm');
    $('editSummary').textContent = `${t.direction === 'in' ? 'دخل' : 'صرف'} ${money(t.amount)} · ${dateFmt.format(new Date(t.occurred_at))}`;
    f.merchant.value = t.merchant || '';
    f.note.value = t.note || '';
    f.category.disabled = t.direction === 'in';
    f.category.value = t.direction === 'in' ? '' : t.category;
    openDialog('editDialog');
  }
  $('editForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = e.target;
    const body = { merchant: f.merchant.value, note: f.note.value };
    if (state.editing.direction === 'out') body.category = f.category.value;
    try {
      await api('PATCH', `/api/transactions/${state.editing.id}`, body);
      $('editDialog').close();
      loadHome();
    } catch (err) {
      toast(err.message);
    }
  });
  $('deleteTx').addEventListener('click', async () => {
    if (!confirm('تحذف هذي العملية؟')) return;
    try {
      await api('DELETE', `/api/transactions/${state.editing.id}`);
      $('editDialog').close();
      toast('تم الحذف');
      loadHome();
    } catch (err) {
      toast(err.message);
    }
  });

  $('prevMonth').addEventListener('click', () => { state.month = shiftMonth(state.month, -1); loadHome(); });
  $('nextMonth').addEventListener('click', () => { state.month = shiftMonth(state.month, 1); loadHome(); });

  // ---------- settings ----------
  let tokenVisible = false;
  function renderToken() {
    $('authValue').textContent = tokenVisible ? `Bearer ${state.token}` : '••••••••';
    $('revealToken').textContent = tokenVisible ? 'إخفاء' : 'إظهار';
  }
  function loadSettings() {
    $('meName').textContent = state.me.name;
    $('smsUrl').textContent = `${location.origin}/api/sms`;
    tokenVisible = false;
    renderToken();
  }
  $('revealToken').addEventListener('click', () => { tokenVisible = !tokenVisible; renderToken(); });

  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-copy]');
    if (!btn) return;
    const id = btn.dataset.copy;
    const text = id === 'authValue' ? `Bearer ${state.token}` : $(id).textContent;
    try {
      await navigator.clipboard.writeText(text);
      toast('تم النسخ');
    } catch {
      toast('ما قدرت أنسخ، انسخه يدوياً');
    }
  });

  $('rotateToken').addEventListener('click', async () => {
    if (!confirm('بيتوقف الرمز القديم فوراً، ولازم تحدّث الاختصار بالرمز الجديد. متأكد؟')) return;
    try {
      const r = await api('POST', '/api/me/token');
      state.token = r.token;
      storeSet(r.token);
      tokenVisible = true;
      renderToken();
      toast('تم تغيير الرمز');
    } catch (err) {
      toast(err.message);
    }
  });

  $('exportCsv').addEventListener('click', async () => {
    try {
      const res = await fetch(`/api/export.csv?month=${state.month}`, { headers: { Authorization: `Bearer ${state.token}` } });
      if (!res.ok) throw new Error('فشل التصدير');
      const url = URL.createObjectURL(await res.blob());
      const a = el('a', { href: url, download: `sarfi-${state.month}.csv` });
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) {
      toast(err.message);
    }
  });

  $('logout').addEventListener('click', () => {
    if (confirm('تسجيل خروج من هذا الجهاز؟')) logout();
  });

  // ---------- owner ----------
  async function loadOwner() {
    try {
      const [inv, mem] = await Promise.all([api('GET', '/api/invites'), api('GET', '/api/members')]);
      $('inviteList').replaceChildren(...inv.invites.map((i) => el('li', {},
        el('span', { text: `${i.note || 'بدون اسم'} · تنتهي ${dateFmt.format(new Date(i.expires_at))}` }),
        el('button', { type: 'button', class: 'small danger', text: 'إلغاء', onclick: () => revokeInvite(i.id) }))));
      if (!inv.invites.length) $('inviteList').replaceChildren(el('li', { class: 'muted', text: 'لا يوجد' }));
      $('memberList').replaceChildren(...mem.members.map((m) => el('li', {},
        el('span', { text: `${m.name}${m.role === 'owner' ? ' (أنت)' : ''} · ${m.tx_count} عملية` }),
        m.role === 'owner' ? null : el('button', { type: 'button', class: 'small danger', text: 'إزالة', onclick: () => removeMember(m) }))));
    } catch (err) {
      toast(err.message);
    }
  }

  $('inviteForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      const r = await api('POST', '/api/invites', { note: e.target.note.value });
      $('inviteLink').textContent = `${location.origin}/#join=${r.code}`;
      $('inviteResult').hidden = false;
      e.target.reset();
      loadOwner();
    } catch (err) {
      toast(err.message);
    }
  });

  async function revokeInvite(id) {
    try {
      await api('DELETE', `/api/invites/${id}`);
      loadOwner();
    } catch (err) {
      toast(err.message);
    }
  }

  async function removeMember(m) {
    if (!confirm(`تشيل ${m.name}؟ بتنحذف كل بياناته ويتوقف دخوله فوراً.`)) return;
    try {
      await api('DELETE', `/api/members/${m.id}`);
      toast('تمت الإزالة');
      loadOwner();
    } catch (err) {
      toast(err.message);
    }
  }

  // ---------- join ----------
  $('joinForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = e.target;
    $('joinError').textContent = '';
    try {
      const r = await api('POST', '/api/join', { code: f.code.value.trim(), name: f.name.value });
      state.token = r.token;
      storeSet(r.token);
      f.reset();
      boot();
    } catch (err) {
      $('joinError').textContent = err.message;
    }
  });

  for (const b of document.querySelectorAll('.tab')) b.addEventListener('click', () => show(b.dataset.view));

  if ('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(() => {});
  boot();
})();
