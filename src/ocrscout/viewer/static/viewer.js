// Browser-side glue for the ocrscout Gradio viewer.
//
// Three responsibilities:
//   1. Synchronized scroll across all `.ocrscout-markdown-pane` panes — when
//      one pane scrolls, mirror the scroll *ratio* (not pixels) to the others
//      so column-fonts/wrapping differences don't drift the alignment.
//   2. Keyboard shortcuts: j/k step pages, [/] toggle sidebar, i toggles the
//      image pane, 1-4 switch view modes, ? shows the help modal.
//   3. Keep the URL query string in sync with the current view so the page
//      can be linked/bookmarked.
//
// The script is loaded once via gr.Blocks(head=...) and idempotently registers
// its handlers on document.body (Gradio re-renders content under the same
// body, so the listeners survive @gr.render passes).

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
      // Release the suppression on the next frame so we don't stack events.
      requestAnimationFrame(() => {
        suppressSync = false;
      });
    },
    true /* capture, since scroll doesn't bubble */
  );

  // --- 2. Keyboard shortcuts ----------------------------------------------

  function clickByLabel(labelText) {
    // Find a Gradio button whose text matches; gradio uses <button>...text...</button>.
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

  // --- 3. URL <-> state sync ----------------------------------------------
  //
  // Exposed as window.ocrscoutSyncUrl so Gradio event handlers can call it
  // from their js= hook with the current state (page, models, mode).
  window.ocrscoutSyncUrl = function (page, models, mode) {
    try {
      const u = new URL(window.location.href);
      if (page) u.searchParams.set("page", page);
      else u.searchParams.delete("page");
      if (models && models.length)
        u.searchParams.set("models", models.join(","));
      else u.searchParams.delete("models");
      if (mode) u.searchParams.set("mode", mode);
      else u.searchParams.delete("mode");
      window.history.replaceState({}, "", u.toString());
    } catch (e) {
      // URL not writable for some reason — silently swallow; not load-bearing.
    }
  };
})();
