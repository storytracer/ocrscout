"""Gradio Blocks app for ocrscout's interactive inspector.

Layout — three independently-collapsible zones:

    ┌─ Sidebar ─────┬─ Image pane ──────────┬─ Text pane ──────────────┐
    │ Page picker   │ AnnotatedImage with   │ Dynamic columns rendered │
    │ Model picker  │ bbox overlay (the     │ via @gr.render — count   │
    │ View mode     │ first selected model  │ matches selected models. │
    │ Stats         │ supplies layout)      │ Mode determines content. │
    └───────────────┴───────────────────────┴──────────────────────────┘

The text pane has three modes (Single / Side-by-side / Diff); the sidebar
and image pane each toggle independently so a user comparing many models
can hide both and get full-bleed text. State (page, models, mode, image
visible) persists in BrowserState; the same fields can also be supplied via
URL query params for shareable links.
"""

from __future__ import annotations

import logging
from html import escape
from pathlib import Path
from typing import Any

import gradio as gr

from ocrscout.viewer.diff import render_diff_table_fragment
from ocrscout.viewer.store import ModelRow, PageRow, ViewerStore

log = logging.getLogger(__name__)

VIEW_MODES = ["Single", "Side-by-side", "Diff"]
STATIC_DIR = Path(__file__).parent / "static"


def build_app(output_dir: Path) -> gr.Blocks:
    """Build the Gradio Blocks app for ``output_dir``.

    Loads the parquet eagerly at startup; image/markdown are resolved on
    demand. Returns the un-launched ``Blocks``; the ``css`` and ``head``
    strings are attached as ``demo.ocrscout_css`` / ``demo.ocrscout_head``
    so the launcher can pass them to ``Blocks.launch(...)`` (gradio 6 API).
    """
    store = ViewerStore(Path(output_dir))

    css = (STATIC_DIR / "viewer.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "viewer.js").read_text(encoding="utf-8")
    head = f"<script>{js}</script>"

    page_choices = store.page_ids()
    if not page_choices:
        # Build a minimal "no data" app rather than crashing — the user can
        # still see what they pointed at.
        with gr.Blocks(title="ocrscout viewer") as demo:
            gr.Markdown(
                f"### No rows in `{output_dir / 'data'}/`.\n\n"
                "Run `ocrscout run` first."
            )
        demo.ocrscout_css = css
        demo.ocrscout_head = head
        return demo

    initial_page = page_choices[0]
    initial_models = store.all_models[: min(2, len(store.all_models))]
    initial_layout_models = store.layout_models_for(initial_page)
    initial_layout_choice = initial_layout_models[0] if initial_layout_models else None

    with gr.Blocks(
        title="ocrscout viewer",
        elem_classes="ocrscout-viewer",
        fill_width=True,
    ) as demo:

        # ----- State -----
        # BrowserState persists across reloads on the same browser; URL query
        # params take precedence on first load via `demo.load(...)` below.
        browser_state = gr.BrowserState(
            {
                "page_id": initial_page,
                "models": initial_models,
                "mode": "Single",
                "show_image": True,
            }
        )
        # Stable, non-persistent state used by event handlers.
        current_page = gr.State(initial_page)

        # ----- Top bar: three logical groups (page nav | view mode | actions)
        with gr.Row(equal_height=True):
            # --- Group 1: page navigation (Prev / Page / Next / Sort) ---
            with gr.Column(scale=5, elem_classes=["ocrscout-control-group"]):
                gr.HTML('<div class="ocrscout-group-label">Page navigation</div>')
                with gr.Row(equal_height=True):
                    with gr.Column(scale=0, min_width=80):
                        prev_btn = gr.Button("‹ Prev", size="sm")
                    with gr.Column(scale=4):
                        page_dd = gr.Dropdown(
                            choices=page_choices,
                            value=initial_page,
                            label=None,
                            show_label=False,
                            container=False,
                            interactive=True,
                            allow_custom_value=False,
                        )
                    with gr.Column(scale=0, min_width=80):
                        next_btn = gr.Button("Next ›", size="sm")

            # --- Group 2: view mode ---
            with gr.Column(scale=4, elem_classes=["ocrscout-control-group"]):
                gr.HTML('<div class="ocrscout-group-label">View mode</div>')
                view_mode = gr.Radio(
                    choices=VIEW_MODES,
                    value="Single",
                    label=None,
                    show_label=False,
                    container=False,
                    elem_id="ocrscout-view-mode",
                )

            # --- Group 3: layout source (which model's bboxes to overlay) ---
            with gr.Column(scale=2, elem_classes=["ocrscout-control-group"]):
                gr.HTML('<div class="ocrscout-group-label">Layout source</div>')
                layout_model_dd = gr.Dropdown(
                    choices=initial_layout_models or [""],
                    value=initial_layout_choice,
                    label=None,
                    show_label=False,
                    container=False,
                )

            # --- Group 4: actions ---
            with gr.Column(scale=2, elem_classes=["ocrscout-control-group"]):
                gr.HTML('<div class="ocrscout-group-label">Actions</div>')
                with gr.Row(equal_height=True):
                    image_toggle = gr.Button("Toggle image", size="sm")
                    help_btn = gr.Button("Help", size="sm")

        # ----- Status strip / page summary -----
        page_summary = gr.HTML(_render_page_summary(store, initial_page))

        # ----- Two-column body: image | text panes -----
        with gr.Row(equal_height=False):
            with gr.Column(
                scale=2,
                visible=True,
                elem_id="ocrscout-image-col",
                elem_classes=["ocrscout-image-col"],
            ) as image_col:
                annotated = gr.AnnotatedImage(
                    label="Source page (boxes from selected layout model)",
                    show_legend=False,
                    color_map=ViewerStore.LABEL_COLORS,
                    elem_id="ocrscout-annotated",
                    elem_classes=["ocrscout-annotated"],
                )
                # Custom dedup'd legend — Gradio's built-in legend renders
                # one chip per annotation (so 12 "text" boxes = 12 chips).
                # We collapse it to one chip per unique label.
                legend_html = gr.HTML(
                    _render_legend([], store),
                    elem_classes=["ocrscout-legend"],
                )
            with gr.Column(scale=5):
                # Selected models live in this State so the picker and the
                # text pane can stay in sync even when the picker rebuilds
                # itself between view-mode changes (radio in Single, checkbox
                # in Side-by-side, paired dropdowns in Diff).
                models_state = gr.State(initial_models)

                @gr.render(
                    inputs=[view_mode, models_state],
                    triggers=[demo.load, view_mode.change],
                )
                def _render_model_picker(mode: str, current: list[str]):
                    _draw_model_picker(store, mode, current, models_state)

                # Re-rendered every time inputs change; column count tracks
                # the number of selected models (or collapses to a single
                # diff table in Diff mode).
                @gr.render(
                    inputs=[current_page, models_state, view_mode],
                    triggers=[
                        demo.load,
                        page_dd.change,
                        models_state.change,
                        view_mode.change,
                    ],
                )
                def _render_text_pane(page_id: str, models: list[str], mode: str):
                    _draw_text_pane(store, page_id, models, mode)

        # ----- Help modal -----
        with gr.Group(visible=False) as help_box:
            gr.HTML(_HELP_HTML)
            help_close = gr.Button("Close", size="sm")

        # ----- Wiring -----

        def _on_page_change(page_id: str) -> tuple:
            summary = _render_page_summary(store, page_id)
            layout_models = store.layout_models_for(page_id)
            layout_choice = layout_models[0] if layout_models else None
            return page_id, summary, gr.update(
                choices=layout_models or [""], value=layout_choice
            )

        page_dd.change(
            _on_page_change,
            inputs=[page_dd],
            outputs=[current_page, page_summary, layout_model_dd],
        )

        def _step(direction: int, page_id: str):
            ids = store.page_ids()
            try:
                idx = ids.index(page_id)
            except ValueError:
                idx = 0
            new_idx = max(0, min(len(ids) - 1, idx + direction))
            new_page = ids[new_idx]
            layout_models = store.layout_models_for(new_page)
            layout_choice = layout_models[0] if layout_models else None
            return (
                gr.update(value=new_page, choices=ids),
                new_page,
                _render_page_summary(store, new_page),
                gr.update(choices=layout_models or [""], value=layout_choice),
            )

        prev_btn.click(
            lambda p: _step(-1, p),
            inputs=[current_page],
            outputs=[page_dd, current_page, page_summary, layout_model_dd],
        )
        next_btn.click(
            lambda p: _step(+1, p),
            inputs=[current_page],
            outputs=[page_dd, current_page, page_summary, layout_model_dd],
        )

        def _on_layout_change(page_id: str, model: str | None):
            # No layout model selected (or none exist for this page) — show
            # the bare source image rather than clearing the pane. Without
            # this, navigating to a layout-less page (e.g. plain markdown
            # OCR like lighton-ocr2 or glm-ocr) blanks the source image.
            if not model:
                img = store.image_for(page_id)
                value = (img, []) if img is not None else None
                return value, _render_legend([], store)
            img, anns = store.annotated_for(page_id, model)
            if img is None:
                return None, _render_legend([], store)
            return (img, anns), _render_legend(anns, store)

        layout_model_dd.change(
            _on_layout_change,
            inputs=[current_page, layout_model_dd],
            outputs=[annotated, legend_html],
        )
        page_dd.change(
            _on_layout_change,
            inputs=[current_page, layout_model_dd],
            outputs=[annotated, legend_html],
        )

        def _toggle_image(visible_state: bool):
            return gr.update(visible=not visible_state), not visible_state

        image_visible_state = gr.State(True)
        image_toggle.click(
            _toggle_image,
            inputs=[image_visible_state],
            outputs=[image_col, image_visible_state],
        )

        # Image stays visible across all view modes; only the user's manual
        # Toggle image button hides it.

        # When the mode changes, normalize the model selection to fit the
        # new mode's constraints so the text pane gets a usable selection
        # without the user having to re-pick.
        def _normalize_models_for_mode(mode: str, current: list[str]) -> list[str]:
            current = list(current or [])
            if mode == "Single":
                return current[:1] or (
                    [store.all_models[0]] if store.all_models else []
                )
            if mode == "Diff":
                seeded = current[:2]
                if len(seeded) < 2:
                    for m in store.all_models:
                        if m not in seeded:
                            seeded.append(m)
                        if len(seeded) >= 2:
                            break
                return seeded[:2]
            # Side-by-side: keep selection, but ensure non-empty.
            return current or store.all_models[: min(2, len(store.all_models))]

        view_mode.change(
            _normalize_models_for_mode,
            inputs=[view_mode, models_state],
            outputs=[models_state],
        )

        help_btn.click(lambda: gr.update(visible=True), outputs=[help_box])
        help_close.click(lambda: gr.update(visible=False), outputs=[help_box])

        # Persist UI state to BrowserState whenever it changes.
        def _save_state(page_id, models, mode, show_image):
            return {
                "page_id": page_id,
                "models": models,
                "mode": mode,
                "show_image": show_image,
            }

        for trigger in (
            page_dd.change,
            models_state.change,
            view_mode.change,
            image_visible_state.change,
        ):
            trigger(
                _save_state,
                inputs=[current_page, models_state, view_mode, image_visible_state],
                outputs=[browser_state],
            )

        # Initial load: read URL query params and BrowserState, settle the UI.
        def _on_load(state: dict[str, Any] | None, request: gr.Request):
            qp = dict(request.query_params) if request else {}
            page_id = qp.get("page") or (state or {}).get("page_id") or initial_page
            if page_id not in page_choices:
                page_id = initial_page
            mode = qp.get("mode") or (state or {}).get("mode") or "Single"
            if mode not in VIEW_MODES:
                mode = "Single"
            models_raw = qp.get("models")
            if models_raw:
                requested = [m.strip() for m in models_raw.split(",") if m.strip()]
                models = [m for m in requested if m in store.all_models]
            else:
                models = (state or {}).get("models") or initial_models
            models = [m for m in models if m in store.all_models]
            if not models and store.all_models:
                models = store.all_models[: min(2, len(store.all_models))]
            layout_models = store.layout_models_for(page_id)
            layout_choice = layout_models[0] if layout_models else None
            img, anns = (
                store.annotated_for(page_id, layout_choice)
                if layout_choice
                else (store.image_for(page_id), [])
            )
            return (
                gr.update(value=page_id, choices=store.page_ids()),
                page_id,
                models,
                gr.update(value=mode),
                gr.update(choices=layout_models or [""], value=layout_choice),
                _render_page_summary(store, page_id),
                (img, anns) if img is not None else None,
                _render_legend(anns, store),
            )

        demo.load(
            _on_load,
            inputs=[browser_state],
            outputs=[
                page_dd,
                current_page,
                models_state,
                view_mode,
                layout_model_dd,
                page_summary,
                annotated,
                legend_html,
            ],
        )

    demo.ocrscout_css = css
    demo.ocrscout_head = head
    return demo


# ---------------------------------------------------------------- helpers


def _render_legend(
    annotations: list[tuple[tuple[int, int, int, int], str]],
    store: ViewerStore,
) -> str:
    """Render a deduplicated colour legend as HTML.

    Gradio's built-in AnnotatedImage legend renders one chip per annotation,
    so 12 ``text`` boxes show as 12 chips. We collapse to one chip per
    unique label, preserving the order in which the labels first appear.
    Each chip's color matches ``ViewerStore.LABEL_COLORS`` (the same map
    fed to ``AnnotatedImage.color_map``), so chips and overlays agree.
    """
    if not annotations:
        return '<div class="ocrscout-legend-empty">no layout for this page</div>'
    seen: dict[str, int] = {}
    for _bbox, label in annotations:
        seen[label] = seen.get(label, 0) + 1
    chips: list[str] = []
    for label, count in seen.items():
        color = store.LABEL_COLORS.get(label, "#888")
        chips.append(
            f'<span class="ocrscout-legend-chip" '
            f'style="--chip-color:{color}">'
            f'<span class="dot" style="background:{color}"></span>'
            f'{escape(label)}<span class="count">×{count}</span>'
            f'</span>'
        )
    return f'<div class="ocrscout-legend-row">{"".join(chips)}</div>'


def _render_page_summary(store: ViewerStore, page_id: str) -> str:
    page: PageRow | None = next(
        (p for p in store.pages() if p.page_id == page_id), None
    )
    if page is None:
        return f'<div class="ocrscout-stats-strip"><span>{escape(page_id)}</span></div>'
    err_html = ""
    if page.error_models:
        err_html = (
            '<span class="err">errors: '
            + escape(", ".join(page.error_models))
            + "</span>"
        )
    return (
        '<div class="ocrscout-stats-strip">'
        f'<span class="name">{escape(page.page_id)}</span>'
        f'<span>models: {len(page.models)}</span>'
        f'<span>disagreement: {page.disagreement:.2f}</span>'
        f'<span>max chars: {page.char_count}</span>'
        f"{err_html}"
        '</div>'
    )


def _draw_model_picker(
    store: ViewerStore, mode: str, current: list[str], models_state: gr.State
) -> None:
    """Emit the model-picker component appropriate to the current view mode.

    * **Single**       — exactly 1 model. ``gr.Radio``.
    * **Side-by-side** — 1..N models. ``gr.CheckboxGroup``.
    * **Diff**         — exactly 2 models. Two paired ``gr.Dropdown`` (A vs B)
      so the user can swap either side without thinking about checkbox order.

    All variants write into ``models_state`` so downstream ``@gr.render``
    blocks (text pane, BrowserState save) only watch one signal.
    """
    all_models = store.all_models

    if mode == "Single":
        value = current[0] if current else (all_models[0] if all_models else None)
        radio = gr.Radio(
            choices=all_models,
            value=value,
            label="Model",
            interactive=True,
            elem_classes=["ocrscout-model-picker"],
        )
        radio.change(
            lambda v: [v] if v else [],
            inputs=[radio],
            outputs=[models_state],
        )
        return

    if mode == "Diff":
        # Seed both sides; if the user has fewer than 2 picked, pad from
        # all_models so neither dropdown starts empty.
        seeded = list(current[:2])
        for m in all_models:
            if len(seeded) >= 2:
                break
            if m not in seeded:
                seeded.append(m)
        a_val = seeded[0] if len(seeded) > 0 else None
        b_val = seeded[1] if len(seeded) > 1 else None
        with gr.Row(equal_height=True):
            dd_a = gr.Dropdown(
                choices=all_models,
                value=a_val,
                label="Model A",
                interactive=True,
                elem_classes=["ocrscout-model-picker"],
            )
            dd_b = gr.Dropdown(
                choices=all_models,
                value=b_val,
                label="Model B",
                interactive=True,
                elem_classes=["ocrscout-model-picker"],
            )

        def _diff_pair(a: str | None, b: str | None) -> list[str]:
            return [m for m in (a, b) if m]

        dd_a.change(_diff_pair, inputs=[dd_a, dd_b], outputs=[models_state])
        dd_b.change(_diff_pair, inputs=[dd_a, dd_b], outputs=[models_state])
        return

    # Side-by-side: free multi-select.
    seeded = list(current) if current else (all_models[: min(2, len(all_models))])
    cg = gr.CheckboxGroup(
        choices=all_models,
        value=seeded,
        label="Models",
        interactive=True,
        elem_classes=["ocrscout-model-picker"],
    )
    cg.change(
        lambda v: list(v) if v else [],
        inputs=[cg],
        outputs=[models_state],
    )


def _draw_text_pane(
    store: ViewerStore, page_id: str, models: list[str], mode: str
) -> None:
    """Body of @gr.render — emits the right component tree for the current mode.

    Called inside Gradio's render context, so it must construct components
    rather than return values.
    """
    if not models:
        gr.Markdown("_Pick at least one model._")
        return

    rows: dict[str, ModelRow] = {}
    for m in models:
        r = store.get(page_id, m)
        if r is not None:
            rows[m] = r
    if not rows:
        gr.Markdown(f"_No model rows for page `{page_id}`._")
        return

    if mode == "Single":
        # Use the first selected model.
        first = next(iter(rows.values()))
        _emit_stats_strip(first)
        _emit_text_body(first, store, use_structured=_has_structure(first))
        return

    if mode == "Diff":
        names = list(rows.keys())
        if len(names) < 2:
            gr.Markdown("_Diff needs exactly two models — pick another._")
            return
        a, b = names[0], names[1]
        if len(names) > 2:
            gr.Markdown(
                f"_Diff uses the first two selected models: **{escape(a)}** vs **{escape(b)}**. "
                "Deselect others to change the pair._"
            )
        fragment, stats = render_diff_table_fragment(
            rows[a].markdown, rows[b].markdown, model_a=a, model_b=b
        )
        gr.HTML(
            f'<div class="ocrscout-stats-strip">'
            f'<span class="name">{escape(a)} ↔ {escape(b)}</span>'
            f'<span>similarity {stats.similarity:.1f}%</span>'
            f'<span>common {stats.common}</span>'
            f'<span>removed {stats.removed}</span>'
            f'<span>added {stats.added}</span>'
            f"</div>"
            + fragment
        )
        return

    # Side-by-side: one column per model.
    n = len(rows)
    # Cap visible columns to avoid pathological sliver layouts. When more
    # than 5 are picked, render the first 5 and surface a hint.
    items = list(rows.items())
    capped = items[:5]
    if n > 5:
        gr.Markdown(
            f"_{n} models selected — showing first 5. Deselect some to see "
            "the rest._"
        )
    # Color-coded sections only when *every* visible row has layout — mixing
    # structured and plain panes side-by-side reads inconsistently. If even
    # one model is plain-markdown, all panes drop back to plain markdown.
    all_structured = all(_has_structure(r) for _, r in capped)
    with gr.Row(equal_height=False):
        for _m, row in capped:
            with gr.Column(scale=1, min_width=200):
                _emit_stats_strip(row)
                _emit_text_body(row, store, use_structured=all_structured)


def _has_structure(row: ModelRow) -> bool:
    """Does this row carry true layout structure worth color-coding?

    True iff the model emitted at least one bbox — i.e. the section labels
    are backed by actual region detection rather than textual heuristics
    (the markdown normalizer assigns ``title``/``section_header``/``table``
    purely from ``#`` and ``<table>`` patterns; those labels exist but
    describe text shape, not detected layout). Color-coded section blocks
    would imply layout when none was performed, so we fall back to plain
    markdown for those rows.
    """
    return len(row.bboxes) > 0


def _emit_text_body(
    row: ModelRow, store: ViewerStore, *, use_structured: bool
) -> None:
    """Render the model's text content for Single / Side-by-side modes.

    When ``use_structured`` is True (Single mode for a layout-aware model,
    or Side-by-side where *all* selected models have layout), render an
    HTML body where each item is a left-bordered block colored to match the
    bbox legend. Otherwise fall back to the flat markdown export — also the
    fallback when even one selected model lacks layout, since mixing
    structured and plain panes side-by-side reads inconsistently.
    """
    if not use_structured or not row.items:
        gr.Markdown(
            row.markdown or "_(empty)_",
            elem_classes="ocrscout-markdown-pane",
        )
        return
    parts = ['<div class="ocrscout-markdown-pane ocrscout-structured">']
    for item in row.items:
        color = store.LABEL_COLORS.get(item.label, "#888")
        # ``item.html`` is pre-rendered HTML (currently TableItems via
        # TableItem.export_to_html) — docling-core escapes cell content,
        # so it's safe to embed raw and the browser renders a real
        # <table>. Otherwise fall through to the plain-text escape path.
        body_html = item.html or escape(item.text or "").replace("\n", "<br>")
        parts.append(
            f'<div class="ocrscout-section" '
            f'data-label="{escape(item.label)}" '
            f'style="--section-color:{color}">'
            f'<span class="ocrscout-section-tag">{escape(item.label)}</span>'
            f'<div class="ocrscout-section-text">{body_html}</div>'
            f'</div>'
        )
    parts.append('</div>')
    gr.HTML("".join(parts))


def _emit_stats_strip(row: ModelRow) -> None:
    metrics = row.metrics or {}
    items = metrics.get("item_count")
    chars = metrics.get("text_length")
    sec = metrics.get("run_seconds_per_page")
    err_html = (
        f'<span class="err">error: {escape(row.error)}</span>' if row.error else ""
    )
    gr.HTML(
        '<div class="ocrscout-stats-strip">'
        f'<span class="name">{escape(row.model)}</span>'
        f'<span>fmt: {escape(row.output_format or "?")}</span>'
        f"<span>items: {items if items is not None else '—'}</span>"
        f"<span>chars: {chars if chars is not None else '—'}</span>"
        f"<span>s/page: {(f'{sec:.2f}' if isinstance(sec, (int, float)) else '—')}</span>"
        f"{err_html}"
        "</div>"
    )


_HELP_HTML = """
<div class="ocrscout-help">
<h3>Keyboard shortcuts</h3>
<dl>
  <dt>j / k</dt><dd>Next / previous page</dd>
  <dt>1 / 2 / 3</dt><dd>Switch to Single / Side-by-side / Diff mode</dd>
  <dt>i</dt><dd>Toggle the image pane</dd>
  <dt>?</dt><dd>Show this help</dd>
</dl>
<h3>View modes</h3>
<dl>
  <dt>Single</dt><dd>One model, full markdown — for reading.</dd>
  <dt>Side-by-side</dt><dd>One column per selected model (caps at 5).
    Synchronized scroll across columns.</dd>
  <dt>Diff</dt><dd>Pairwise word-level diff between two models. Pick A and B
    from the dropdowns.</dd>
</dl>
<h3>URL parameters</h3>
<p>Append <code>?page=...&amp;models=a,b&amp;mode=Diff</code> to share a
specific view.</p>
</div>
"""
