// Browser-side glue for the ocrscout Gradio viewer.
//
// Four responsibilities:
//   1. Synchronized scroll across all `.ocrscout-markdown-pane` panes.
//   2. Keyboard shortcuts: j/k step pages, i toggles the image pane,
//      1-3 switch view modes, ? help.
//   3. Diff toggles: bind the split/unified mode buttons, the
//      "changes only" checkbox, and the minimap click-to-jump on every
//      .ocrscout-diff that mounts.
//   4. URL sync: keep ?file=, ?models=, ?mode= in the URL bar so views
//      are shareable / bookmarkable.

(function () {
  if (window.__ocrscoutViewerLoaded) return;
  window.__ocrscoutViewerLoaded = true;

  // --- 1. Synchronized scroll ----------------------------------------------

  let suppressSync = false;
  document.body.addEventListener(
    "scroll",
    function (ev) {
      const target = ev.target;
      if (
        !target ||
        !target.classList ||
        !target.classList.contains("ocrscout-markdown-pane")
      ) {
        return;
      }
      if (suppressSync) return;
      const denom = target.scrollHeight - target.clientHeight;
      const ratio = denom > 0 ? target.scrollTop / denom : 0;
      const panes = document.querySelectorAll(".ocrscout-markdown-pane");
      suppressSync = true;
      panes.forEach((p) => {
        if (p === target) return;
        const d = p.scrollHeight - p.clientHeight;
        if (d > 0) p.scrollTop = ratio * d;
      });
      requestAnimationFrame(() => {
        suppressSync = false;
      });
    },
    true
  );

  // --- 2. Keyboard shortcuts -----------------------------------------------

  function clickByLabel(labelText) {
    const buttons = document.querySelectorAll("button");
    for (const b of buttons) {
      if ((b.textContent || "").trim() === labelText) {
        b.click();
        return true;
      }
    }
    return false;
  }

  function setRadioValueByLabel(elemId, label) {
    const root = document.getElementById(elemId);
    if (!root) return false;
    const inputs = root.querySelectorAll("input[type=radio]");
    for (const inp of inputs) {
      const lbl = inp.closest("label");
      if (lbl && (lbl.textContent || "").trim().startsWith(label)) {
        inp.click();
        return true;
      }
    }
    return false;
  }

  function isTypingTarget(el) {
    if (!el) return false;
    const tag = (el.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return true;
    if (el.isContentEditable) return true;
    return false;
  }

  document.addEventListener("keydown", function (ev) {
    if (isTypingTarget(ev.target)) return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    switch (ev.key) {
      case "j":
        clickByLabel("Next ›");
        ev.preventDefault();
        break;
      case "k":
        clickByLabel("‹ Prev");
        ev.preventDefault();
        break;
      case "i":
        clickByLabel("Toggle image");
        ev.preventDefault();
        break;
      case "1":
        setRadioValueByLabel("ocrscout-view-mode", "Single");
        ev.preventDefault();
        break;
      case "2":
        setRadioValueByLabel("ocrscout-view-mode", "Side-by-side");
        ev.preventDefault();
        break;
      case "3":
        setRadioValueByLabel("ocrscout-view-mode", "Compare");
        ev.preventDefault();
        break;
      case "?":
        clickByLabel("Help");
        ev.preventDefault();
        break;
    }
  });

  // --- 3. Diff toggles -----------------------------------------------------

  function bindDiff(root) {
    root.querySelectorAll(".ocrscout-diff").forEach((diff) => {
      if (diff.dataset.bound === "1") return;
      diff.dataset.bound = "1";

      diff.querySelectorAll(".diff-mode .mode-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          const mode = btn.dataset.mode;
          diff.dataset.mode = mode;
          diff.querySelectorAll(".diff-mode .mode-btn").forEach((b) => {
            b.classList.toggle("active", b === btn);
          });
        });
      });

      diff.querySelectorAll(".changes-only-toggle").forEach((toggle) => {
        toggle.addEventListener("change", () => {
          diff.dataset.changesOnly = toggle.checked ? "1" : "0";
        });
      });

      diff.querySelectorAll(".ocrscout-diff-minimap").forEach((mm) => {
        mm.addEventListener("click", (ev) => {
          const pane = diff.querySelector(".ocrscout-diff-pane");
          if (!pane) return;
          const rect = mm.getBoundingClientRect();
          const ratio = (ev.clientY - rect.top) / rect.height;
          pane.scrollTo({
            top: ratio * pane.scrollHeight,
            behavior: "smooth",
          });
        });
      });
    });
  }
  window.ocrscoutBindDiff = (root) => bindDiff(root || document);

  // --- 4. URL <-> state sync ----------------------------------------------

  window.ocrscoutSyncUrl = function (file, models, mode) {
    try {
      const u = new URL(window.location.href);
      // Migrate legacy ?page= to ?file=.
      u.searchParams.delete("page");
      if (file) u.searchParams.set("file", file);
      else u.searchParams.delete("file");
      if (models && models.length)
        u.searchParams.set("models", models.join(","));
      else u.searchParams.delete("models");
      if (mode) u.searchParams.set("mode", mode);
      else u.searchParams.delete("mode");
      window.history.replaceState({}, "", u.toString());
    } catch (e) {
      // URL not writable — silently swallow.
    }
  };

  // --- Periodic re-bind to catch Gradio re-renders -------------------------

  function rebindAll() {
    bindDiff(document);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", rebindAll);
  } else {
    rebindAll();
  }
  // Gradio swaps content under the same body element; observing for
  // additions catches re-renders without a polling loop.
  const observer = new MutationObserver(() => rebindAll());
  observer.observe(document.body, { childList: true, subtree: true });
})();
