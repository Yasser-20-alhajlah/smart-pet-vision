"""Browser page for the live demonstration, served by serve.py at GET /demo.

Presentation layer only: the page sends the chosen file to POST /analyze and displays the JSON it
returns. It adds no decision logic, changes no threshold and needs no network access (no external
scripts, styles or fonts). Display rules applied here and not in the API:
- for a `no_animal` referral the classifier still returns a species for the whole frame, so the page
  shows neither that species nor its confidence as an identification;
- for any other referral the classifier species is labelled provisional and the reasons come first;
- a high score (including 1.000) is shown as a model score, not as proof that the species is correct.
The page keeps nothing: no browser storage, no cookies, and the original image is shown from a
browser object URL that is released when the result is cleared or replaced.
"""

DEMO_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Pet Vision - photo review demo</title>
<style>
  :root { --ink:#1d2430; --muted:#535d6c; --line:#d5dbe3; --bg:#f4f6f9; --card:#ffffff;
          --ok:#185c32; --okbg:#e3f3e8; --warn:#7a4200; --warnbg:#fff1dc; --bad:#8f1a1a; --badbg:#fde4e4;
          --accent:#1f5fae; --focus:#c25e00; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif; color:var(--ink); background:var(--bg); }
  header { background:var(--card); border-bottom:1px solid var(--line); padding:14px 16px; }
  header h1 { margin:0; font-size:20px; line-height:1.3; }
  header p { margin:4px 0 0; color:var(--muted); font-size:13px; }
  main { max-width:1180px; margin:0 auto; padding:16px 16px 40px; }
  h2 { font-size:17px; margin:0 0 8px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:14px; }
  .policy { display:flex; flex-wrap:wrap; gap:6px 20px; font-weight:600; }
  .policy code { font-weight:400; color:var(--muted); }
  .limits { font-size:13px; color:var(--muted); margin:8px 0 0; }
  #drop { display:block; position:relative; border:2px dashed #8f9bad; border-radius:10px; padding:18px 12px; text-align:center; background:#fbfcfe; cursor:pointer; }
  #drop.over { border-color:var(--accent); background:#eef4fc; }
  #drop:focus-within { outline:3px solid var(--focus); outline-offset:2px; }
  /* Visually hidden but focusable, so the file chooser is reachable with the keyboard. */
  #file { position:absolute; width:1px; height:1px; opacity:0; overflow:hidden; clip:rect(0 0 0 0); }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:12px; }
  button { font:inherit; font-weight:600; padding:9px 18px; border-radius:8px; border:1px solid var(--accent); background:var(--accent); color:#fff; cursor:pointer; min-height:40px; }
  button.secondary { background:#fff; color:var(--accent); }
  button:disabled { opacity:.5; cursor:not-allowed; }
  button:focus-visible { outline:3px solid var(--focus); outline-offset:2px; }
  #fileinfo { color:var(--muted); font-size:13px; overflow-wrap:anywhere; }
  #status { min-height:22px; font-size:14px; flex:1 1 220px; }
  #status.error { color:var(--bad); font-weight:600; }
  .spinner { display:inline-block; width:14px; height:14px; border:2px solid #b9c4d4; border-top-color:var(--accent); border-radius:50%; animation:spin .8s linear infinite; vertical-align:-2px; margin-right:6px; }
  @keyframes spin { to { transform:rotate(360deg); } }
  @media (prefers-reduced-motion: reduce) { .spinner { animation:none; } }
  #verdict { border-radius:10px; padding:14px 16px; margin-bottom:14px; border:2px solid; }
  #verdict:focus { outline:3px solid var(--focus); outline-offset:2px; }
  #verdict.ready { background:var(--okbg); border-color:var(--ok); color:var(--ok); }
  #verdict.refer { background:var(--warnbg); border-color:var(--warn); color:var(--warn); }
  #verdict.alert { background:var(--badbg); border-color:var(--bad); color:var(--bad); }
  #verdict h2 { margin:0 0 4px; font-size:21px; line-height:1.3; }
  #verdict p { margin:0; color:var(--ink); }
  #verdict ul { margin:8px 0 0 20px; padding:0; color:var(--ink); }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }
  figure { margin:0; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px; }
  figure img { width:100%; height:250px; object-fit:contain; background:repeating-conic-gradient(#eef1f5 0 25%,#fff 0 50%) 0 0/16px 16px; border-radius:6px; display:block; }
  figure .na { height:250px; display:flex; align-items:center; justify-content:center; text-align:center; color:var(--muted); background:#f0f2f6; border-radius:6px; padding:12px; }
  figcaption { font-size:13px; color:var(--muted); margin-top:6px; }
  figcaption strong { color:var(--ink); }
  .tablewrap { overflow-x:auto; }
  table { border-collapse:collapse; width:100%; font-size:14px; }
  th, td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { color:var(--muted); font-weight:600; }
  .kv { display:grid; grid-template-columns:minmax(150px,max-content) 1fr; gap:6px 16px; font-size:14px; }
  .kv dt { color:var(--muted); }
  .kv dd { margin:0; overflow-wrap:anywhere; }
  .note { font-size:13px; color:var(--muted); }
  details pre { max-height:320px; overflow:auto; background:#0f1722; color:#e6edf3; padding:10px; border-radius:8px; font-size:12px; }
  .hidden { display:none !important; }
  footer { color:var(--muted); font-size:12px; margin-top:18px; }
  @media (max-width:560px) {
    .kv { grid-template-columns:1fr; gap:0 0; }
    .kv dd { margin-bottom:8px; }
    #verdict h2 { font-size:18px; }
    button { flex:1 1 auto; }
  }
</style>
</head>
<body>
<header>
  <h1>Smart Pet Vision &mdash; photo review demo</h1>
  <p>Detector, classifier and segmenter run behind <code>POST /analyze</code>. Every result is a proposal for shelter staff, never a publication decision.</p>
</header>
<main>
  <section class="card" aria-labelledby="policy-title">
    <h2 id="policy-title">Publication policy (fixed in the service)</h2>
    <div class="policy">
      <span>Staff approval required for every photo <code>staff_approval_required = true</code></span>
      <span>Automatic publication disabled <code>automatic_publication_allowed = false</code></span>
    </div>
    <p class="limits">Limits: the models were trained on Oxford-IIIT Pet photos (one cat or dog per photo). On other photo types, with several animals or with small animals, results can be wrong even when scores are high.</p>
  </section>

  <section class="card" aria-labelledby="upload-title">
    <h2 id="upload-title">1. Choose a photo</h2>
    <label id="drop" for="file">
      <input id="file" type="file" accept="image/*" aria-describedby="fileinfo">
      <strong>Choose a photo</strong> or drop it here<br>
      <span id="fileinfo">JPEG or PNG, up to 15 MB and 25 megapixels</span>
    </label>
    <div class="row">
      <button id="analyze" type="button" disabled>Analyse photo</button>
      <button id="clear" type="button" class="secondary" disabled>Clear</button>
      <div id="status" role="status" aria-live="polite"></div>
    </div>
  </section>

  <section id="results" class="hidden" aria-labelledby="result-title">
    <h2 id="result-title" class="hidden">2. Result</h2>
    <div id="verdict" tabindex="-1"></div>
    <div class="grid" id="images"></div>
    <div class="card" style="margin-top:14px">
      <dl class="kv" id="facts"></dl>
    </div>
    <div class="card" id="detcard">
      <strong>Detections</strong> <span class="note">(detector operating point from the release manifest; scores are model scores, not proof)</span>
      <div class="tablewrap">
        <table id="dets"><thead><tr><th scope="col">#</th><th scope="col">Species</th><th scope="col">Detector score</th><th scope="col">Box x1, y1, x2, y2 (pixels)</th></tr></thead><tbody></tbody></table>
      </div>
    </div>
    <details class="card"><summary>Raw service response (JSON, images omitted; for technical review)</summary><p id="rawnote" class="note hidden">For a no_animal referral the species and confidence fields below come from the whole frame; they do not identify an animal.</p><pre id="raw"></pre></details>
  </section>

  <footer>Presentation layer over the service API: no decision logic and no approval action in this page.
    The service does not write uploaded images to disk and this page stores nothing in the browser.
    Label curated and synthetic example images as such when presenting.</footer>
</main>
<script>
(function () {
  "use strict";
  var MAX_BYTES = 15 * 1024 * 1024;
  var VERDICTS = { "ready": true, "refer_to_staff": true };
  var fileInput = document.getElementById("file");
  var drop = document.getElementById("drop");
  var analyzeBtn = document.getElementById("analyze");
  var clearBtn = document.getElementById("clear");
  var statusBox = document.getElementById("status");
  var info = document.getElementById("fileinfo");
  var results = document.getElementById("results");
  var verdictBox = document.getElementById("verdict");
  var chosen = null;        // the File currently selected
  var originalUrl = null;   // object URL of the shown original, released when no longer needed
  var request = null;       // {id, controller, timer, tick} of the request in flight
  var requestSeq = 0;       // responses from an older request are ignored

  function el(tag, text, cls) {
    var e = document.createElement(tag);
    if (text !== undefined && text !== null) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }
  function setStatus(text, isError, busy) {
    statusBox.textContent = "";
    statusBox.className = isError ? "error" : "";
    if (busy) statusBox.appendChild(el("span", null, "spinner"));
    statusBox.appendChild(document.createTextNode(text));
  }
  function releaseOriginal() {
    if (originalUrl) { URL.revokeObjectURL(originalUrl); originalUrl = null; }
  }
  function clearResults() {
    // Remove every trace of the previous result so nothing stale stays on screen.
    results.classList.add("hidden");
    verdictBox.textContent = "";
    verdictBox.className = "";
    document.getElementById("images").textContent = "";
    document.getElementById("facts").textContent = "";
    document.querySelector("#dets tbody").textContent = "";
    document.getElementById("raw").textContent = "";
    document.getElementById("rawnote").classList.add("hidden");
    releaseOriginal();
  }
  function cancelRequest() {
    if (!request) return;
    clearTimeout(request.timer);
    clearInterval(request.tick);
    request.controller.abort();
    request = null;
  }
  function updateButtons() {
    var busy = request !== null;
    analyzeBtn.disabled = busy || !chosen || chosen.size === 0 || chosen.size > MAX_BYTES;
    clearBtn.disabled = !busy && !chosen && results.classList.contains("hidden") && statusBox.className !== "error";
    analyzeBtn.textContent = busy ? "Analysing..." : "Analyse photo";
  }
  function choose(file) {
    cancelRequest();
    clearResults();
    chosen = file || null;
    if (!chosen) {
      info.textContent = "JPEG or PNG, up to 15 MB and 25 megapixels";
      setStatus("", false);
    } else {
      info.textContent = "Selected: " + chosen.name + " (" + (chosen.size / 1024 / 1024).toFixed(2) + " MB)";
      if (chosen.size === 0) setStatus("This file is empty. Choose another photo.", true);
      else if (chosen.size > MAX_BYTES) setStatus("This file is larger than 15 MB, the service limit. Choose a smaller photo.", true);
      else setStatus("Ready to analyse.", false);
    }
    updateButtons();
  }
  function resetAll() {
    cancelRequest();
    clearResults();
    chosen = null;
    fileInput.value = "";
    info.textContent = "JPEG or PNG, up to 15 MB and 25 megapixels";
    setStatus("Cleared. Choose a photo to start again.", false);
    updateButtons();
    fileInput.focus();
  }

  fileInput.addEventListener("change", function () { choose(fileInput.files[0]); });
  ["dragenter", "dragover"].forEach(function (t) { drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
  ["dragleave", "drop"].forEach(function (t) { drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
  drop.addEventListener("drop", function (e) { if (e.dataTransfer && e.dataTransfer.files.length) choose(e.dataTransfer.files[0]); });
  clearBtn.addEventListener("click", resetAll);

  var HTTP_MESSAGES = {
    400: "The upload was empty.",
    413: "The image is too large for the service (over 15 MB, or more than 25 megapixels once decoded).",
    415: "The file is not a readable image.",
    422: "No file was received by the service.",
    503: "The service could not produce a valid model output for this photo. Staff review required; no result is shown."
  };

  function fail(message) {
    clearResults();
    setStatus(message, true);
    updateButtons();
  }

  analyzeBtn.addEventListener("click", function () {
    if (!chosen || request) return;  // one request at a time
    var file = chosen;
    var form = new FormData();
    form.append("file", file, file.name);
    clearResults();
    var id = ++requestSeq;
    var controller = new AbortController();
    var started = performance.now();
    request = { id: id, controller: controller,
                timer: setTimeout(function () { controller.abort(); }, 180000),
                tick: setInterval(function () {
                  setStatus("Analysing... " + Math.round((performance.now() - started) / 1000) + " s (the first request after start-up also loads the models).", false, true);
                }, 1000) };
    setStatus("Analysing... the first request after start-up also loads the models and can take longer.", false, true);
    updateButtons();
    fetch("/analyze?images=true", { method: "POST", body: form, signal: controller.signal })
      .then(function (resp) {
        return resp.text().then(function (body) {
          var data = null;
          try { data = JSON.parse(body); } catch (e) { data = null; }
          if (!resp.ok) {
            var detail = data && typeof data.detail === "string" ? " Service message: " + data.detail : "";
            var err = new Error("HTTP " + resp.status + ": " + (HTTP_MESSAGES[resp.status] || "The service returned an error.") + detail);
            err.http = true;
            throw err;
          }
          if (!data || typeof data !== "object" || !VERDICTS[data.verdict] || !Array.isArray(data.reasons)) {
            var bad = new Error("The service returned an unexpected response, so no result is shown. Staff review required.");
            bad.http = true;
            throw bad;
          }
          return data;
        });
      })
      .then(function (data) {
        if (!request || request.id !== id || chosen !== file) return;  // superseded or cleared
        var seconds = (performance.now() - started) / 1000;
        finish();
        try {
          render(data, seconds, file);
          setStatus("Done in " + seconds.toFixed(1) + " s.", false);
        } catch (e) {
          fail("The response could not be displayed, so no result is shown. Staff review required.");
        }
      })
      .catch(function (err) {
        if (!request || request.id !== id) return;  // superseded or cleared: say nothing
        var timedOut = err && err.name === "AbortError";
        finish();
        fail(timedOut ? "No response within 3 minutes. Check that the service is running, then try again."
             : (err && err.http ? err.message
             : "The service could not be reached. Check that uvicorn is running on this address, then try again."));
      });

    function finish() {
      clearTimeout(request.timer);
      clearInterval(request.tick);
      request = null;
      updateButtons();
    }
  });

  function figure(title, src, caption, naText) {
    var f = el("figure");
    if (src) { var img = el("img"); img.alt = title + ": " + caption; img.src = src; f.appendChild(img); }
    else f.appendChild(el("div", naText, "na"));
    var c = el("figcaption"); c.appendChild(el("strong", title)); c.appendChild(document.createTextNode(" — " + caption));
    f.appendChild(c);
    return f;
  }
  function score(x) { return Number(x).toFixed(3); }

  function render(r, seconds, file) {
    var reasons = r.reasons;
    var codes = reasons.map(function (x) { return x && x.code; });
    var noAnimal = codes.indexOf("no_animal") !== -1;
    var policyOk = r.staff_approval_required === true && r.automatic_publication_allowed === false;
    var ready = policyOk && r.verdict === "ready" && reasons.length === 0;

    verdictBox.textContent = "";
    if (!policyOk) {
      verdictBox.className = "alert";
      verdictBox.appendChild(el("h2", "Unexpected policy flags — do not use this result"));
      verdictBox.appendChild(el("p", "staff_approval_required = " + r.staff_approval_required + ", automatic_publication_allowed = " + r.automatic_publication_allowed));
    } else if (ready) {
      verdictBox.className = "ready";
      verdictBox.appendChild(el("h2", "Ready for staff review — not approved for publication"));
      verdictBox.appendChild(el("p", "No referral reason was raised. A staff member must still check and approve the photo before anything is published."));
    } else {
      verdictBox.className = "refer";
      verdictBox.appendChild(el("h2", "Staff review required — see reasons"));
      verdictBox.appendChild(el("p", reasons.length === 1 ? "Reason raised by the service:" : reasons.length + " reasons raised by the service:"));
      var ul = el("ul");
      reasons.forEach(function (x) {
        var li = el("li");
        li.appendChild(el("strong", String(x && x.code)));
        li.appendChild(document.createTextNode(": " + String(x && x.message)));
        ul.appendChild(li);
      });
      verdictBox.appendChild(ul);
    }

    releaseOriginal();
    originalUrl = URL.createObjectURL(file);
    var grid = document.getElementById("images");
    var png = function (b64) { return typeof b64 === "string" && b64 ? "data:image/png;base64," + b64 : null; };
    grid.appendChild(figure("Original", originalUrl, "the photo as uploaded (shown from this browser)"));
    if (noAnimal) {
      grid.appendChild(figure("Detections and mask", null, "nothing to draw", "No animal was detected, so there are no boxes and no mask."));
      grid.appendChild(figure("Mask", null, "not applicable", "Not applicable: no animal detected."));
      grid.appendChild(figure("Cutout", null, "not applicable", "Not applicable: no animal detected."));
    } else {
      grid.appendChild(figure("Detections and mask", png(r.annotated_png), "detector boxes (orange) and segmenter mask (blue) on the photo", "No overlay image was returned."));
      grid.appendChild(figure("Mask", png(r.mask_png), "white = pixels the segmenter assigned to the selected animal", "No mask image was returned."));
      grid.appendChild(figure("Cutout", png(r.cutout_png), "selected animal with the background removed", "No cutout image was returned."));
    }

    var facts = document.getElementById("facts");
    var add = function (k, val) { facts.appendChild(el("dt", k)); facts.appendChild(el("dd", val)); };
    var dets = Array.isArray(r.detections) ? r.detections : [];
    if (noAnimal) {
      add("Identification", "Not shown: no animal was detected. The classifier still ran on the whole frame, so its species and score do not identify an animal.");
    } else {
      var label = ready ? "Species (classifier)" : "Species (classifier, provisional)";
      var text = String(r.species) + " — score " + score(r.confidence) + ". A model score, temperature-scaled on Oxford validation photos; even 1.000 is not proof that the species is correct.";
      if (!ready) text += " Check the reasons above before relying on it.";
      add(label, text);
      add("Animals detected", String(dets.length) + (dets.length > 1 ? " (more than one animal: referred)" : ""));
      add("Mask matched to the selected animal", r.segmentation_matched === true ? "yes" : "no");
      add("Mask share of the image", (100 * Number(r.mask_fraction)).toFixed(1) + " %");
    }
    add("Staff approval required", r.staff_approval_required === true ? "yes (staff_approval_required = true)" : String(r.staff_approval_required));
    add("Automatic publication allowed", r.automatic_publication_allowed === false ? "no (automatic_publication_allowed = false)" : String(r.automatic_publication_allowed));
    if (Array.isArray(r.image_size)) add("Image size", r.image_size[0] + " x " + r.image_size[1] + " px");
    var total = r.timings && r.timings.total_ms;
    add("Time", (typeof total === "number" ? total.toFixed(0) + " ms in the service, " : "") + seconds.toFixed(1) + " s round trip on this machine (a functional check, not a benchmark)");

    var tbody = document.querySelector("#dets tbody");
    document.getElementById("detcard").classList.toggle("hidden", dets.length === 0);
    dets.forEach(function (d, i) {
      var tr = el("tr");
      tr.appendChild(el("td", String(i + 1)));
      tr.appendChild(el("td", String(d.species)));
      tr.appendChild(el("td", score(d.confidence)));
      tr.appendChild(el("td", Array.isArray(d.box_xyxy) ? d.box_xyxy.map(function (x) { return Math.round(x); }).join(", ") : ""));
      tbody.appendChild(tr);
    });

    var copy = {};
    Object.keys(r).forEach(function (k) { if (!/_png$/.test(k)) copy[k] = r[k]; });
    document.getElementById("raw").textContent = JSON.stringify(copy, null, 2);
    document.getElementById("rawnote").classList.toggle("hidden", !noAnimal);
    document.getElementById("result-title").classList.remove("hidden");
    results.classList.remove("hidden");
    updateButtons();
    verdictBox.focus();
  }
})();
</script>
</body>
</html>
"""
