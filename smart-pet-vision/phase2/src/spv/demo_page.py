"""Browser page for the live demonstration, served by serve.py at GET /demo.

Presentation layer only: the page sends the chosen file to POST /analyze and displays the JSON it
returns. It adds no decision logic, changes no threshold and needs no network access (no external
scripts, styles or fonts). One display rule is applied here and not in the API: for a `no_animal`
referral the classifier still returns a species for the whole frame, so the page does not show that
species or its confidence as an identification.
"""

DEMO_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Pet Vision - photo review demo</title>
<style>
  :root { --ink:#1d2430; --muted:#5b6575; --line:#d9dee6; --bg:#f5f7fa; --card:#ffffff;
          --ok:#1d6b3a; --okbg:#e3f3e8; --warn:#8a4b00; --warnbg:#fff1dc; --bad:#9b1c1c; --badbg:#fde4e4;
          --accent:#1f5fae; }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif; color:var(--ink); background:var(--bg); }
  header { background:var(--card); border-bottom:1px solid var(--line); padding:14px 20px; }
  header h1 { margin:0; font-size:20px; }
  header p { margin:2px 0 0; color:var(--muted); font-size:13px; }
  main { max-width:1180px; margin:0 auto; padding:16px 20px 40px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:14px; }
  .policy { display:flex; flex-wrap:wrap; gap:8px 18px; font-weight:600; }
  .policy span { white-space:nowrap; }
  #drop { display:block; border:2px dashed #aab4c3; border-radius:10px; padding:18px; text-align:center; background:#fbfcfe; cursor:pointer; }
  #drop.over { border-color:var(--accent); background:#eef4fc; }
  #drop input { display:none; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:10px; }
  button { font:inherit; font-weight:600; padding:9px 18px; border-radius:8px; border:1px solid var(--accent); background:var(--accent); color:#fff; cursor:pointer; }
  button:disabled { opacity:.55; cursor:default; }
  #fileinfo { color:var(--muted); font-size:13px; }
  #status { min-height:22px; font-size:14px; }
  #status.error { color:var(--bad); font-weight:600; }
  .spinner { display:inline-block; width:14px; height:14px; border:2px solid #b9c4d4; border-top-color:var(--accent); border-radius:50%; animation:spin .8s linear infinite; vertical-align:-2px; margin-right:6px; }
  @keyframes spin { to { transform:rotate(360deg); } }
  #verdict { border-radius:10px; padding:14px 16px; margin-bottom:14px; border:2px solid; }
  #verdict.ready { background:var(--okbg); border-color:var(--ok); color:var(--ok); }
  #verdict.refer { background:var(--warnbg); border-color:var(--warn); color:var(--warn); }
  #verdict.alert { background:var(--badbg); border-color:var(--bad); color:var(--bad); }
  #verdict h2 { margin:0 0 4px; font-size:22px; letter-spacing:.3px; }
  #verdict ul { margin:6px 0 0 18px; padding:0; color:var(--ink); }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:12px; }
  figure { margin:0; background:var(--card); border:1px solid var(--line); border-radius:10px; padding:8px; }
  figure img { width:100%; height:260px; object-fit:contain; background:repeating-conic-gradient(#eef1f5 0 25%,#fff 0 50%) 0 0/16px 16px; border-radius:6px; display:block; }
  figure .na { height:260px; display:flex; align-items:center; justify-content:center; text-align:center; color:var(--muted); background:#f0f2f6; border-radius:6px; padding:12px; }
  figcaption { font-size:13px; color:var(--muted); margin-top:6px; }
  table { border-collapse:collapse; width:100%; font-size:14px; }
  th, td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; }
  .kv { display:grid; grid-template-columns:max-content 1fr; gap:4px 14px; font-size:14px; }
  .kv div:nth-child(odd) { color:var(--muted); }
  details pre { max-height:320px; overflow:auto; background:#0f1722; color:#e6edf3; padding:10px; border-radius:8px; font-size:12px; }
  .hidden { display:none; }
  footer { color:var(--muted); font-size:12px; margin-top:18px; }
</style>
</head>
<body>
<header>
  <h1>Smart Pet Vision &mdash; photo review demo</h1>
  <p>Detector, classifier and segmenter run behind <code>POST /analyze</code>. The result is a proposal for shelter staff, never a publication decision.</p>
</header>
<main>
  <section class="card policy" aria-label="Publication policy">
    <span>&#128100; Staff approval required for every photo</span>
    <span>&#9940; Automatic publication disabled</span>
  </section>

  <section class="card" aria-label="Upload">
    <label id="drop" for="file">
      <input id="file" type="file" accept="image/*">
      <strong>Choose a photo</strong> or drop it here<br>
      <span id="fileinfo">JPEG or PNG, up to 15 MB and 25 megapixels</span>
    </label>
    <div class="row">
      <button id="analyze" type="button" disabled>Analyse photo</button>
      <div id="status" role="status" aria-live="polite"></div>
    </div>
  </section>

  <section id="results" class="hidden" aria-label="Result">
    <div id="verdict"></div>
    <div class="grid" id="images"></div>
    <div class="card" style="margin-top:14px">
      <div class="kv" id="facts"></div>
    </div>
    <div class="card" id="detcard">
      <strong>Detections</strong> (detector operating point from the release manifest)
      <table id="dets"><thead><tr><th>#</th><th>Species</th><th>Score</th><th>Box (x1, y1, x2, y2), pixels</th></tr></thead><tbody></tbody></table>
    </div>
    <details class="card"><summary>Response JSON (images omitted)</summary><pre id="raw"></pre></details>
  </section>

  <footer>Presentation layer over the service API: no decision logic in this page. The service does not write uploaded images to disk.
    Label curated and synthetic example images as such when presenting.</footer>
</main>
<script>
(function () {
  "use strict";
  var MAX_BYTES = 15 * 1024 * 1024;
  var fileInput = document.getElementById("file");
  var drop = document.getElementById("drop");
  var button = document.getElementById("analyze");
  var statusBox = document.getElementById("status");
  var info = document.getElementById("fileinfo");
  var results = document.getElementById("results");
  var chosen = null, originalUrl = null;

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
  function choose(file) {
    chosen = file || null;
    results.classList.add("hidden");
    if (!chosen) { button.disabled = true; info.textContent = "JPEG or PNG, up to 15 MB and 25 megapixels"; return; }
    info.textContent = chosen.name + " (" + (chosen.size / 1024 / 1024).toFixed(2) + " MB)";
    if (chosen.size === 0) { button.disabled = true; setStatus("This file is empty. Choose another photo.", true); return; }
    if (chosen.size > MAX_BYTES) { button.disabled = true; setStatus("This file is larger than 15 MB, the service limit. Choose a smaller photo.", true); return; }
    button.disabled = false;
    setStatus("Ready to analyse.", false);
  }
  fileInput.addEventListener("change", function () { choose(fileInput.files[0]); });
  ["dragenter", "dragover"].forEach(function (t) { drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
  ["dragleave", "drop"].forEach(function (t) { drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
  drop.addEventListener("drop", function (e) { if (e.dataTransfer.files.length) choose(e.dataTransfer.files[0]); });

  var HTTP_MESSAGES = {
    400: "The upload was empty.",
    413: "The image is too large (over 15 MB or 25 megapixels).",
    415: "The file is not a readable image.",
    422: "No file was received by the service.",
    503: "The model output was unavailable or invalid; the photo needs staff review."
  };

  button.addEventListener("click", function () {
    if (!chosen) return;
    var form = new FormData();
    form.append("file", chosen, chosen.name);
    button.disabled = true;
    results.classList.add("hidden");
    setStatus("Analysing... the first request after start-up also loads the models and can take longer.", false, true);
    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, 180000);
    var started = performance.now();
    fetch("/analyze?images=true", { method: "POST", body: form, signal: controller.signal })
      .then(function (resp) {
        return resp.text().then(function (body) {
          var data = null;
          try { data = JSON.parse(body); } catch (e) { data = null; }
          if (!resp.ok) {
            var detail = data && typeof data.detail === "string" ? " (" + data.detail + ")" : "";
            throw new Error("HTTP " + resp.status + ": " + (HTTP_MESSAGES[resp.status] || "The service returned an error.") + detail);
          }
          if (!data) throw new Error("The service returned a response that is not JSON.");
          return data;
        });
      })
      .then(function (data) {
        render(data, (performance.now() - started) / 1000);
        setStatus("Done.", false);
      })
      .catch(function (err) {
        var msg = err && err.name === "AbortError" ? "No response within 3 minutes. Check that the service is running, then try again."
                : (err && err.message && err.message.indexOf("HTTP ") === 0 ? err.message
                : "The service could not be reached. Check that uvicorn is running on this address.");
        setStatus(msg, true);
      })
      .then(function () { clearTimeout(timer); button.disabled = !chosen; });
  });

  function figure(title, src, caption, naText) {
    var f = el("figure");
    if (src) { var img = el("img"); img.alt = title; img.src = src; f.appendChild(img); }
    else f.appendChild(el("div", naText, "na"));
    var c = el("figcaption"); c.appendChild(el("strong", title)); c.appendChild(document.createTextNode(" - " + caption));
    f.appendChild(c);
    return f;
  }

  function render(r, seconds) {
    var reasons = Array.isArray(r.reasons) ? r.reasons : [];
    var codes = reasons.map(function (x) { return x.code; });
    var noAnimal = codes.indexOf("no_animal") !== -1;
    var policyOk = r.staff_approval_required === true && r.automatic_publication_allowed === false;

    var v = document.getElementById("verdict");
    v.textContent = "";
    if (!policyOk) {
      v.className = "alert";
      v.appendChild(el("h2", "UNEXPECTED POLICY FLAGS - do not use this result"));
      v.appendChild(el("div", "staff_approval_required=" + r.staff_approval_required + ", automatic_publication_allowed=" + r.automatic_publication_allowed));
    } else if (r.verdict === "ready") {
      v.className = "ready";
      v.appendChild(el("h2", "READY CANDIDATE"));
      v.appendChild(el("div", "No referral reason was raised. A staff member must still approve the photo before it is published."));
    } else {
      v.className = "refer";
      v.appendChild(el("h2", "REFER TO STAFF"));
      v.appendChild(el("div", "The photo needs a staff decision for the following reason(s):"));
      var ul = el("ul");
      reasons.forEach(function (x) { var li = el("li"); li.appendChild(el("strong", x.code)); li.appendChild(document.createTextNode(": " + x.message)); ul.appendChild(li); });
      v.appendChild(ul);
    }

    if (originalUrl) URL.revokeObjectURL(originalUrl);
    originalUrl = URL.createObjectURL(chosen);
    var grid = document.getElementById("images");
    grid.textContent = "";
    var png = function (b64) { return b64 ? "data:image/png;base64," + b64 : null; };
    grid.appendChild(figure("Original", originalUrl, "the uploaded photo"));
    if (noAnimal) {
      grid.appendChild(figure("Detections and mask", null, "nothing to draw", "No animal was detected, so there are no boxes and no mask."));
      grid.appendChild(figure("Mask", null, "not applicable", "Not applicable: no animal detected."));
      grid.appendChild(figure("Cutout", null, "not applicable", "Not applicable: no animal detected."));
    } else {
      grid.appendChild(figure("Detections and mask", png(r.annotated_png), "boxes in orange, mask in blue"));
      grid.appendChild(figure("Mask", png(r.mask_png), "white = animal pixels"));
      grid.appendChild(figure("Cutout", png(r.cutout_png), "selected animal, background removed"));
    }

    var facts = document.getElementById("facts");
    facts.textContent = "";
    var add = function (k, val) { facts.appendChild(el("div", k)); facts.appendChild(el("div", val)); };
    var dets = Array.isArray(r.detections) ? r.detections : [];
    if (noAnimal) {
      add("Identification", "Not shown: no animal was detected. The classifier still ran on the whole frame, so its species and confidence do not identify an animal.");
    } else {
      add("Species (classifier)", r.species + "  (confidence " + Number(r.confidence).toFixed(3) + ", temperature-scaled; not calibrated on new photo types)");
      add("Animals detected", String(dets.length) + (dets.length > 1 ? " (more than one animal: referred)" : ""));
      add("Mask matched to the selected animal", r.segmentation_matched ? "yes" : "no");
      add("Mask share of the image", (100 * Number(r.mask_fraction)).toFixed(1) + " %");
    }
    add("Staff approval required", r.staff_approval_required === true ? "yes" : String(r.staff_approval_required));
    add("Automatic publication allowed", r.automatic_publication_allowed === false ? "no" : String(r.automatic_publication_allowed));
    if (r.image_size) add("Image size", r.image_size[0] + " x " + r.image_size[1] + " px");
    var total = r.timings && r.timings.total_ms;
    add("Service time", (typeof total === "number" ? total.toFixed(0) + " ms in the service, " : "") + seconds.toFixed(1) + " s round trip (this machine; a functional check, not a benchmark)");

    var tbody = document.querySelector("#dets tbody");
    tbody.textContent = "";
    document.getElementById("detcard").classList.toggle("hidden", dets.length === 0);
    dets.forEach(function (d, i) {
      var tr = el("tr");
      tr.appendChild(el("td", String(i + 1)));
      tr.appendChild(el("td", d.species));
      tr.appendChild(el("td", Number(d.confidence).toFixed(3)));
      tr.appendChild(el("td", d.box_xyxy.map(function (x) { return Math.round(x); }).join(", ")));
      tbody.appendChild(tr);
    });

    var copy = {};
    Object.keys(r).forEach(function (k) { if (!/_png$/.test(k)) copy[k] = r[k]; });
    document.getElementById("raw").textContent = JSON.stringify(copy, null, 2);
    results.classList.remove("hidden");
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }
})();
</script>
</body>
</html>
"""
