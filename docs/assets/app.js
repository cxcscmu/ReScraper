/* ReScraper project page — case viewer */
(function () {
  "use strict";
  var CASES = window.CASES || [];
  var OPS = ["all", "keep", "edit", "delete", "rewrite"];
  var OPCOLOR = { keep: "var(--keep)", edit: "var(--edit)", delete: "var(--delete)", rewrite: "var(--rewrite)" };
  var state = { filter: "all", idx: 0, showResiliparse: false };

  var $ = function (id) { return document.getElementById(id); };
  var esc = function (s) {
    return String(s == null ? "" : s).replace(/[&<>]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c];
    });
  };
  var pretty = function (s) { return String(s || "").replace(/_/g, " "); };
  var num = function (x, d) { return x == null ? "—" : Number(x).toFixed(d == null ? 2 : d); };

  function list() {
    return state.filter === "all" ? CASES : CASES.filter(function (c) { return c.op === state.filter; });
  }

  /* ---- theme ---- */
  var root = document.documentElement;
  function isDark() {
    var cur = root.getAttribute("data-theme");
    return cur ? cur === "dark"
      : !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }
  function paintToggle() {
    var dark = isDark();
    $("theme-ico").textContent = dark ? "\u2600" : "\u263E";
    $("theme-txt").textContent = dark ? "Light" : "Dark";
    $("theme").setAttribute("aria-pressed", String(dark));
    $("theme").title = dark ? "Switch to the light theme" : "Switch to the dark theme";
  }
  try {
    var saved = localStorage.getItem("rs-theme");
    if (saved) root.setAttribute("data-theme", saved);
  } catch (e) { /* storage may be unavailable */ }
  $("theme").addEventListener("click", function () {
    var next = isDark() ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("rs-theme", next); } catch (e) { /* ignore */ }
    paintToggle();
  });
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    if (mq.addEventListener) mq.addEventListener("change", paintToggle);
  }
  paintToggle();

  /* ---- chips ---- */
  var chips = $("chips");
  OPS.forEach(function (op) {
    var n = op === "all" ? CASES.length : CASES.filter(function (c) { return c.op === op; }).length;
    var b = document.createElement("button");
    b.className = "chip";
    b.type = "button";
    b.setAttribute("aria-pressed", String(op === state.filter));
    b.dataset.op = op;
    b.innerHTML = (op === "all" ? "All cases" : esc(op)) + " <span style=\"opacity:.65\">" + n + "</span>";
    b.addEventListener("click", function () {
      var cur = list()[state.idx];
      state.filter = op;
      var l = list();
      var keep = cur ? l.findIndex(function (c) { return c.gid === cur.gid; }) : -1;
      state.idx = keep >= 0 ? keep : 0;
      render();
    });
    chips.appendChild(b);
  });

  /* ---- panels ---- */
  function panelHTML(o, ours) {
    var dropped = !o.t || !o.t.trim();
    var pill = ours ? o.status : (dropped ? "dropped" : "kept");
    var pillClass = ours ? o.status : (dropped ? "dropped" : "kept");
    var meta = [];
    if (!ours && !dropped && o.status && o.status !== "kept") meta.push(pretty(o.status));
    if (!dropped) meta.push(o.words + " words");
    if (o.dataman != null) meta.push("DataMan " + num(o.dataman, 1));
    if (o.edu != null) meta.push("Edu " + num(o.edu, 2));
    var body = dropped
      ? "<p class=\"ptext\"><span class=\"empty\">" +
        (ours ? "the page is removed from the corpus" : "nothing kept — the page is dropped") + "</span></p>"
      : "<pre class=\"ptext\">" + esc(o.t) + (o.cut ? "\n\n[… truncated for the web page]" : "") + "</pre>";
    var program = ours && o.program
      ? "<div class=\"program\">" + esc(o.program) + "</div>" : "";
    var note = !ours && o.note && dropped
      ? "<div class=\"pfoot\">dropped by: " + esc(pretty(o.note).split("\n")[0]) + "</div>" : "";
    return "<div class=\"panel" + (ours ? " ours" : "") + "\">" +
      "<div class=\"phead\"><span class=\"name\">" + esc(o.label) + "</span>" +
      "<span class=\"pill " + esc(pillClass) + "\">" + esc(pill) + "</span>" +
      "<span class=\"meta\">" + meta.join(" · ") + "</span></div>" +
      program + body + note + "</div>";
  }

  function sourceHTML(c) {
    var showR = state.showResiliparse;
    var text = showR ? c.resiliparse : c.source;
    var label = showR ? "resiliparse text · what every baseline reads"
                      : "rendered page · what ReScraper reads";
    var lines = (text.t || "").split("\n").length;
    var meta = [lines + " lines"];
    if (!showR && c.pre.dataman != null) meta.push("DataMan " + num(c.pre.dataman, 1) + " before cleaning");
    if (!showR && c.pre.edu != null) meta.push("Edu " + num(c.pre.edu, 2));
    var body = (text.t || "").trim()
      ? "<pre class=\"ptext\">" + esc(text.t).replace(/&lt;lid:(\d+)&gt;/g,
          "<span class=\"lid\">&lt;lid:$1&gt;</span>") +
        (text.cut ? "\n\n[… truncated for the web page]" : "") + "</pre>"
      : "<p class=\"ptext\"><span class=\"empty\">the scraper returned nothing</span></p>";
    return "<div class=\"phead\"><span class=\"name\">Source</span>" +
      "<span class=\"meta\">" + meta.join(" · ") + "</span></div>" +
      "<div class=\"program\" style=\"color:var(--muted);background:transparent\">" + esc(label) + "</div>" +
      body +
      "<div class=\"pfoot\"><button type=\"button\" id=\"togglesrc\">" +
      (showR ? "show the rendered page" : "show the resiliparse text") + "</button>" +
      "<span>baselines clean the resiliparse text; ReScraper starts from the rendered page</span></div>";
  }

  /* ---- render ---- */
  function render() {
    var l = list();
    if (!l.length) return;
    if (state.idx >= l.length) state.idx = l.length - 1;
    var c = l[state.idx];

    Array.prototype.forEach.call(chips.children, function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.op === state.filter));
    });

    var track = $("track");
    track.innerHTML = "";
    l.forEach(function (cc, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.style.setProperty("--dotc", OPCOLOR[cc.op] || "var(--muted)");
      b.title = "Case " + (i + 1) + " · " + cc.op + " · " + cc.domain;
      if (i === state.idx) b.setAttribute("aria-current", "true");
      b.addEventListener("click", function () { state.idx = i; render(); });
      track.appendChild(b);
    });

    $("counter").innerHTML = "Case <b>" + (state.idx + 1) + "</b> / " + l.length;
    $("prev").disabled = state.idx === 0;
    $("next").disabled = state.idx === l.length - 1;

    $("c-domain").textContent = c.domain;
    var a = $("c-url");
    a.href = c.url;
    a.textContent = c.url.length > 96 ? c.url.slice(0, 96) + "…" : c.url;

    var faith = c.faith
      ? " Rewrite faithfulness: BERTScore-F1 " + c.faith.bs_F.toFixed(3) +
        ", " + c.faith.src_words + " → " + c.faith.rw_words + " words, no invented entity or number."
      : "";
    $("c-verdict").innerHTML = "<b>Judge:</b> " + esc(c.judge.verdict) + ", informative value " +
      c.judge.value + "/3 — " + esc(c.judge.reason) + esc(faith);

    $("p-source").innerHTML = sourceHTML(c);
    $("togglesrc").addEventListener("click", function () {
      state.showResiliparse = !state.showResiliparse;
      render();
    });

    // left column: the rule stacks, under the page itself; right column: the model-based
    // refiners with ReScraper first
    var RULES = ["refinedweb_rule", "fineweb_rule"];
    var isRule = function (o) { return RULES.indexOf(o.key) >= 0; };
    var isOurs = function (o) { return o.key === "rescraper"; };
    var models = c.out.filter(isOurs).concat(c.out.filter(function (o) {
      return !isOurs(o) && !isRule(o);
    }));
    $("p-rules").innerHTML = c.out.filter(isRule).map(function (o) {
      return panelHTML(o, false);
    }).join("");
    $("p-outputs").innerHTML = models.map(function (o) {
      return panelHTML(o, isOurs(o));
    }).join("");

    if (location.hash !== "#case-" + c.gid) {
      history.replaceState(null, "", "#case-" + c.gid);
    }
  }

  function move(d) {
    var l = list();
    state.idx = Math.min(l.length - 1, Math.max(0, state.idx + d));
    render();
  }
  $("prev").addEventListener("click", function () { move(-1); });
  $("next").addEventListener("click", function () { move(1); });
  document.addEventListener("keydown", function (e) {
    if (e.target && /input|textarea/i.test(e.target.tagName)) return;
    if (e.key === "ArrowLeft") move(-1);
    if (e.key === "ArrowRight") move(1);
  });

  var m = /^#case-(\d+)$/.exec(location.hash || "");
  if (m) {
    var i = CASES.findIndex(function (c) { return String(c.gid) === m[1]; });
    if (i >= 0) state.idx = i;
  }
  render();
})();
