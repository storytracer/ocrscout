"""Gradio Blocks app for ocrscout's interactive inspector.

Top bar carries page navigation (Prev / typeable Dropdown / Next), view
mode, layout-source picker, and the image-toggle action. Below that, a
volume header (when the source has volume metadata) and a two-column
body: image pane and text panes. The text pane has three modes (Single
/ Side-by-side / Compare) and the image pane stays visible across all
of them. State persists in BrowserState; URL query params override on
first load.
"""

from __future__ import annotations

import logging
from html import escape
from pathlib import Path
from typing import Any

import gradio as gr
from gradio_image_annotation_redaction import image_annotator

from ocrscout import registry as _registry
from ocrscout.interfaces.comparison import BaselineView, PredictionView
from ocrscout.viewer.store import BaselineRow, ModelRow, PageRow, ViewerStore

log = logging.getLogger(__name__)

VIEW_MODES = ["Single", "Side-by-side", "Compare"]
# Pseudo-model token for the page's reference baseline.
REFERENCE_PSEUDO_MODEL = "reference"
STATIC_DIR = Path(__file__).parent / "static"

# Layout-source dropdown opt-out. Bbox overlay rendering is expensive on
# large documents; the default is the empty-string sentinel so pages load
# without annotations until the user explicitly picks a layout.
LAYOUT_NONE_VALUE = ""


def _layout_choices(models: list[str]) -> list[tuple[str, str]]:
    """Layout-source dropdown choices: None opt-out + available models."""
    return [("None", LAYOUT_NONE_VALUE)] + [(m, m) for m in models]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` into an ``(r, g, b)`` int tuple for the bbox
    overlay component, which expects per-box colour as RGB."""
    s = hex_color.lstrip("#")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _default_layout_value(layout_models: list[str]) -> str:
    """Default layout-source selection for a page: pick the first available
    layout model when present, else fall back to the None opt-out."""
    return layout_models[0] if layout_models else LAYOUT_NONE_VALUE


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

    file_choices = store.file_ids()
    if not file_choices:
        with gr.Blocks(title="ocrscout viewer") as demo:
            gr.Markdown(
                f"### No rows in `{output_dir / 'data'}/`.\n\n"
                "Run `ocrscout run` first."
            )
        demo.ocrscout_css = css
        demo.ocrscout_head = head
        return demo

    initial_file = file_choices[0]
    # Side-by-side defaults to ALL models — user requested this so that the
    # full set is visible by default and the user un-checks what they don't
    # want, rather than the inverse. The CSS grid wraps responsively.
    initial_models = list(store.all_models)
    initial_layout_models = store.layout_models_for(initial_file)
    initial_layout_choice = _default_layout_value(initial_layout_models)

    with gr.Blocks(
        title="ocrscout viewer",
        elem_classes="ocrscout-viewer",
        fill_width=True,
    ) as demo:

        # ----- State -----
        browser_state = gr.BrowserState(
            {
                "file_id": initial_file,
                "models": initial_models,
                "mode": "Single",
                "show_image": True,
            }
        )
        current_file = gr.State(initial_file)

        # ----- Top bar: page nav + view mode + layout source + actions -----
        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_classes=["ocrscout-control-group"]):
                gr.HTML('<div class="ocrscout-group-label">Page</div>')
                with gr.Row(equal_height=True):
                    with gr.Column(scale=0, min_width=80):
                        prev_btn = gr.Button("‹ Prev", size="sm")
                    with gr.Column(scale=4):
                        page_dd = gr.Dropdown(
                            choices=file_choices,
                            value=initial_file,
                            label=None,
                            show_label=False,
                            container=False,
                            interactive=True,
                            allow_custom_value=False,
                            filterable=True,
                        )
                    with gr.Column(scale=0, min_width=80):
                        next_btn = gr.Button("Next ›", size="sm")
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
            with gr.Column(scale=2, elem_classes=["ocrscout-control-group"]):
                gr.HTML('<div class="ocrscout-group-label">Layout source</div>')
                layout_model_dd = gr.Dropdown(
                    choices=_layout_choices(initial_layout_models),
                    value=initial_layout_choice,
                    label=None,
                    show_label=False,
                    container=False,
                )
            with gr.Column(scale=2, elem_classes=["ocrscout-control-group"]):
                gr.HTML('<div class="ocrscout-group-label">Actions</div>')
                with gr.Row(equal_height=True):
                    image_toggle = gr.Button("Toggle image", size="sm")
                    help_btn = gr.Button("Help", size="sm")

        # ----- Volume / page header -----
        page_summary = gr.HTML(_render_page_summary(store, initial_file))

        # ----- Two-column body: image | text panes -----
        with gr.Row(equal_height=False):
            with gr.Column(
                scale=2,
                visible=True,
                elem_id="ocrscout-image-col",
                elem_classes=["ocrscout-image-col"],
            ) as image_col:
                annotated = image_annotator(
                    label="Source page (boxes from selected layout model)",
                    interactive=False,
                    disable_edit_boxes=True,
                    show_clear_button=False,
                    show_remove_button=False,
                    show_share_button=False,
                    show_download_button=False,
                    enable_keyboard_shortcuts=False,
                    handles_cursor=False,
                    # Hide the per-corner resize handles. Boxes are
                    # display-only here so the handles add visual noise
                    # without function. Setting size=0 makes the canvas
                    # draw a 0x0 rect per handle (a no-op).
                    handle_size=0,
                    sources=[],
                    elem_id="ocrscout-annotated",
                    elem_classes=["ocrscout-annotated"],
                )
                legend_html = gr.HTML(
                    _render_legend([], store),
                    elem_classes=["ocrscout-legend"],
                )
            with gr.Column(scale=5):
                models_state = gr.State(initial_models)

                @gr.render(
                    inputs=[view_mode, models_state],
                    triggers=[demo.load, view_mode.change],
                )
                def _render_model_picker(mode: str, current: list[str]):
                    _draw_model_picker(store, mode, current, models_state)

                @gr.render(
                    inputs=[current_file, models_state, view_mode],
                    triggers=[
                        demo.load,
                        current_file.change,
                        models_state.change,
                        view_mode.change,
                    ],
                )
                def _render_text_pane(file_id: str, models: list[str], mode: str):
                    _draw_text_pane(store, file_id, models, mode)

        # ----- Help modal -----
        with gr.Group(visible=False) as help_box:
            gr.HTML(_HELP_HTML)
            help_close = gr.Button("Close", size="sm")

        # ----- Wiring -----

        def _image_outputs(file_id: str, model: str | None):
            """Compute the (annotated_value, legend_html) pair for a given
            (file_id, layout_model). Single source of truth for image-column
            updates — invoked inline from navigation handlers and from the
            layout dropdown's change listener.

            Returns the dict shape that ``image_annotator`` expects:
            ``{"image": <PIL|URL>, "boxes": [{xmin, ymin, xmax, ymax,
            label, color}, …]}``.
            """
            if not model:
                img = store.image_for(file_id)
                if img is None:
                    return None, _render_legend([], store)
                return {"image": img, "boxes": []}, _render_legend([], store)
            img, anns = store.annotated_for(file_id, model)
            if img is None:
                return None, _render_legend([], store)
            # Omit ``label`` from each box dict — the upstream Canvas
            # renders an in-image label chip whenever ``box.label`` is
            # non-empty, and the legend strip below the image already
            # carries that information. ``color`` is per-box so the
            # category-colour mapping survives.
            boxes = [
                {
                    "xmin": int(x1), "ymin": int(y1),
                    "xmax": int(x2), "ymax": int(y2),
                    "color": _hex_to_rgb(
                        store.LABEL_COLORS.get(label, "#888888")
                    ),
                }
                for (x1, y1, x2, y2), label in anns
            ]
            return {"image": img, "boxes": boxes}, _render_legend(anns, store)

        def _on_file_change(file_id: str) -> tuple:
            summary = _render_page_summary(store, file_id)
            layout_models = store.layout_models_for(file_id)
            layout_value = _default_layout_value(layout_models)
            annotated_value, legend = _image_outputs(file_id, layout_value)
            return (
                file_id,
                summary,
                gr.update(
                    choices=_layout_choices(layout_models),
                    value=layout_value,
                ),
                annotated_value,
                legend,
            )

        page_dd.change(
            _on_file_change,
            inputs=[page_dd],
            outputs=[
                current_file,
                page_summary,
                layout_model_dd,
                annotated,
                legend_html,
            ],
        )

        def _step(direction: int, file_id: str):
            ids = store.file_ids()
            try:
                idx = ids.index(file_id)
            except ValueError:
                idx = 0
            new_idx = max(0, min(len(ids) - 1, idx + direction))
            new_file = ids[new_idx]
            layout_models = store.layout_models_for(new_file)
            layout_value = _default_layout_value(layout_models)
            annotated_value, legend = _image_outputs(new_file, layout_value)
            return (
                gr.update(value=new_file, choices=ids),
                new_file,
                _render_page_summary(store, new_file),
                gr.update(
                    choices=_layout_choices(layout_models),
                    value=layout_value,
                ),
                annotated_value,
                legend,
            )

        prev_btn.click(
            lambda p: _step(-1, p),
            inputs=[current_file],
            outputs=[
                page_dd,
                current_file,
                page_summary,
                layout_model_dd,
                annotated,
                legend_html,
            ],
        )
        next_btn.click(
            lambda p: _step(+1, p),
            inputs=[current_file],
            outputs=[
                page_dd,
                current_file,
                page_summary,
                layout_model_dd,
                annotated,
                legend_html,
            ],
        )

        # Image-column updates land via the navigation handlers above; the
        # layout dropdown still drives its own refresh when the user picks a
        # different overlay model on the same page.
        layout_model_dd.change(
            _image_outputs,
            inputs=[current_file, layout_model_dd],
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

        def _normalize_models_for_mode(mode: str, current: list[str]) -> list[str]:
            current = list(current or [])
            if mode == "Single":
                return current[:1] or (
                    [store.all_models[0]] if store.all_models else []
                )
            if mode == "Compare":
                seeded = current[:2]
                if len(seeded) < 1 and store.all_models:
                    seeded.append(store.all_models[0])
                # Default the baseline (slot 1) to the reference whenever one
                # is available — typical workflow is "compare prediction
                # against reference." A user who picks a different baseline
                # via the dropdown keeps that selection until they next
                # toggle view mode.
                if store.has_any_baseline():
                    if len(seeded) < 2:
                        seeded.append(REFERENCE_PSEUDO_MODEL)
                    else:
                        seeded[1] = REFERENCE_PSEUDO_MODEL
                elif len(seeded) < 2:
                    fallback = next(
                        (m for m in store.all_models if m not in seeded), None
                    )
                    if fallback:
                        seeded.append(fallback)
                return seeded[:2]
            # Side-by-side: default to ALL models if nothing is selected.
            return current or list(store.all_models)

        view_mode.change(
            _normalize_models_for_mode,
            inputs=[view_mode, models_state],
            outputs=[models_state],
        )

        help_btn.click(lambda: gr.update(visible=True), outputs=[help_box])
        help_close.click(lambda: gr.update(visible=False), outputs=[help_box])

        def _save_state(file_id, models, mode, show_image):
            return {
                "file_id": file_id,
                "models": models,
                "mode": mode,
                "show_image": show_image,
            }

        for trigger in (
            current_file.change,
            models_state.change,
            view_mode.change,
            image_visible_state.change,
        ):
            trigger(
                _save_state,
                inputs=[current_file, models_state, view_mode, image_visible_state],
                outputs=[browser_state],
            )

        # Initial load: read URL query params and BrowserState, settle the UI.
        def _on_load(state: dict[str, Any] | None, request: gr.Request):
            qp = dict(request.query_params) if request else {}
            # Accept both `?file=` (new) and `?page=` (legacy) for back-compat.
            requested_id = (
                qp.get("file") or qp.get("page")
                or (state or {}).get("file_id")
                or (state or {}).get("page_id")
                or initial_file
            )
            if requested_id not in file_choices:
                requested_id = initial_file
            mode = qp.get("mode") or (state or {}).get("mode") or "Single"
            if mode not in VIEW_MODES:
                mode = "Single"
            models_raw = qp.get("models")
            if models_raw:
                requested = [m.strip() for m in models_raw.split(",") if m.strip()]
                models = [m for m in requested if m in store.all_models]
                explicit_models = True
            else:
                models = (state or {}).get("models") or initial_models
                explicit_models = False
            models = [m for m in models if m in store.all_models]
            if not models and store.all_models:
                models = list(store.all_models)
            # Compare mode: default the baseline (slot 1) to the reference
            # when one is available, unless the URL pinned an explicit set.
            if (
                mode == "Compare"
                and not explicit_models
                and store.has_any_baseline()
                and store.all_models
            ):
                pred = models[0] if models else store.all_models[0]
                models = [pred, REFERENCE_PSEUDO_MODEL]
            layout_models = store.layout_models_for(requested_id)
            layout_value = _default_layout_value(layout_models)
            annotated_value, legend = _image_outputs(
                requested_id, layout_value
            )
            return (
                gr.update(value=requested_id, choices=store.file_ids()),
                requested_id,
                models,
                gr.update(value=mode),
                gr.update(
                    choices=_layout_choices(layout_models),
                    value=layout_value,
                ),
                _render_page_summary(store, requested_id),
                annotated_value,
                legend,
            )

        demo.load(
            _on_load,
            inputs=[browser_state],
            outputs=[
                page_dd,
                current_file,
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
            f"{escape(label)}<span class=\"count\">×{count}</span>"
            f"</span>"
        )
    return f'<div class="ocrscout-legend-row">{"".join(chips)}</div>'


def _render_page_summary(store: ViewerStore, file_id: str) -> str:
    """Top-of-page header. When the page has volume context, shows
    ``Volume <id> · <title> (<year>) — <author>`` plus ``Page <seq> of N``;
    otherwise just the file_id and basic stats."""
    page: PageRow | None = next(
        (p for p in store.pages() if p.file_id == file_id), None
    )
    if page is None:
        return (
            f'<div class="ocrscout-stats-strip"><span>{escape(file_id)}</span></div>'
        )
    err_html = ""
    if page.error_models:
        err_html = (
            '<span class="err">errors: '
            + escape(", ".join(page.error_models))
            + "</span>"
        )

    volume = store.volume_for(file_id)
    volume_lines: list[str] = []
    if volume is not None:
        title_bits: list[str] = []
        if volume.title:
            title_bits.append(escape(volume.title))
        if volume.year:
            title_bits.append(f"({volume.year})")
        if volume.creators:
            authors = ", ".join(volume.creators[:3])
            if authors:
                title_bits.append(f"— {escape(authors)}")
        head = (
            f'<div class="vol-head">'
            f'<span class="vol-id">Volume {escape(volume.barcode)}</span>'
            + (
                ' · <span class="vol-title">' + " ".join(title_bits) + "</span>"
                if title_bits else ""
            )
            + (
                f' <a class="vol-link" href="{escape(volume.source_uri)}" '
                f'target="_blank" rel="noopener">↗</a>'
                if volume.source_uri
                else ""
            )
            + "</div>"
        )
        volume_lines.append(head)
        seq_str = (
            f"Page {page.sequence}"
            + (f" of {volume.page_count}" if volume.page_count else "")
            if page.sequence is not None
            else ""
        )
        volume_lines.append(
            '<div class="vol-page">'
            + (f"<span>{seq_str}</span>" if seq_str else "")
            + f' <span class="vol-fid">{escape(file_id)}</span>'
            + "</div>"
        )

    if volume_lines:
        return (
            '<div class="ocrscout-stats-strip ocrscout-volume-header">'
            + "".join(volume_lines)
            + '<div class="vol-meta">'
            f'<span>models: {len(page.models)}</span>'
            f'<span>disagreement: {page.disagreement:.2f}</span>'
            f'<span>max chars: {page.char_count}</span>'
            f"{err_html}"
            "</div>"
            "</div>"
        )

    return (
        '<div class="ocrscout-stats-strip">'
        f'<span class="name">{escape(page.file_id)}</span>'
        f'<span>models: {len(page.models)}</span>'
        f'<span>disagreement: {page.disagreement:.2f}</span>'
        f'<span>max chars: {page.char_count}</span>'
        f"{err_html}"
        "</div>"
    )


def _draw_model_picker(
    store: ViewerStore, mode: str, current: list[str], models_state: gr.State
) -> None:
    """Emit the model-picker component appropriate to the current view mode."""
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

    if mode == "Compare":
        baseline_choices = (
            [REFERENCE_PSEUDO_MODEL] if store.has_any_baseline() else []
        )
        a_choices = list(all_models)
        # Reference at the top of the baseline dropdown — typical workflow
        # is to compare prediction against the page's reference.
        b_choices = baseline_choices + list(all_models)
        seeded = list(current[:2])
        # Treat the picker's input as "default fill" (the user hasn't
        # actively picked a baseline yet) when current carries the full
        # all-models list, or when slot 1 is missing/duplicates slot 0. In
        # those cases force the baseline to ``reference`` if available.
        is_default_fill = (
            list(current) == list(all_models)
            or len(current) > 2
        )
        if len(seeded) < 1 and a_choices:
            seeded.append(a_choices[0])
        if baseline_choices and (
            is_default_fill
            or len(seeded) < 2
            or (len(seeded) >= 2 and seeded[1] == seeded[0])
        ):
            if len(seeded) < 2:
                seeded.append(REFERENCE_PSEUDO_MODEL)
            else:
                seeded[1] = REFERENCE_PSEUDO_MODEL
        elif len(seeded) < 2:
            seeded.append(
                next((m for m in all_models if m not in seeded), None)
            )
        a_val = seeded[0] if len(seeded) > 0 else None
        b_val = seeded[1] if len(seeded) > 1 else None
        # Baseline left, prediction right — matches the diff renderer's
        # column orientation (baseline = removed/red on the left,
        # prediction = added/green on the right).
        with gr.Row(equal_height=True):
            dd_b = gr.Dropdown(
                choices=b_choices,
                value=b_val,
                label="Baseline",
                interactive=True,
                elem_classes=["ocrscout-model-picker"],
            )
            dd_a = gr.Dropdown(
                choices=a_choices,
                value=a_val,
                label="Prediction",
                interactive=True,
                elem_classes=["ocrscout-model-picker"],
            )

        def _compare_pair(a: str | None, b: str | None) -> list[str]:
            return [m for m in (a, b) if m]

        dd_a.change(_compare_pair, inputs=[dd_a, dd_b], outputs=[models_state])
        dd_b.change(_compare_pair, inputs=[dd_a, dd_b], outputs=[models_state])
        return

    # Side-by-side: free multi-select. All models default-checked per user
    # request. The grid CSS wraps responsively so 8 panes flow into multiple
    # rows on a narrow viewport.
    sidebyside_choices = list(all_models) + (
        [REFERENCE_PSEUDO_MODEL] if store.has_any_baseline() else []
    )
    seeded = list(current) if current else list(sidebyside_choices)
    cg = gr.CheckboxGroup(
        choices=sidebyside_choices,
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
    store: ViewerStore, file_id: str, models: list[str], mode: str
) -> None:
    """Body of @gr.render — emits the right component tree for the current mode."""
    if not models:
        gr.Markdown("_Pick at least one model._")
        return

    if mode == "Compare":
        _draw_compare_pane(store, file_id, models)
        return

    columns: list[tuple[str, ModelRow | BaselineRow]] = []
    for m in models:
        if m == REFERENCE_PSEUDO_MODEL:
            baselines = store.baselines_for(file_id)
            if baselines:
                columns.append((m, baselines[0]))
        else:
            r = store.get(file_id, m)
            if r is not None:
                columns.append((m, r))
    if not columns:
        gr.Markdown(f"_No content for `{escape(file_id)}` with the current selection._")
        return

    if mode == "Single":
        _label, row = columns[0]
        if isinstance(row, BaselineRow):
            _emit_baseline_stats_strip(row)
            _emit_baseline_text_body(row)
        else:
            _emit_stats_strip(row)
            _emit_text_body(row, store, use_structured=_has_structure(row))
        return

    # Side-by-side: one column per selected artifact. No hard cap; CSS grid
    # handles wrapping when there are many panes.
    all_structured = all(
        isinstance(row, ModelRow) and _has_structure(row)
        for _, row in columns
    )
    gr.HTML('<div class="ocrscout-side-by-side-grid">')
    with gr.Row(equal_height=False, elem_classes=["ocrscout-sbs-row"]):
        for _label, row in columns:
            with gr.Column(scale=1, min_width=320):
                if isinstance(row, BaselineRow):
                    _emit_baseline_stats_strip(row)
                    _emit_baseline_text_body(row)
                else:
                    _emit_stats_strip(row)
                    _emit_text_body(row, store, use_structured=all_structured)
    gr.HTML("</div>")


def _has_structure(row: ModelRow) -> bool:
    return len(row.bboxes) > 0


def _draw_compare_pane(
    store: ViewerStore, file_id: str, models: list[str]
) -> None:
    if len(models) < 2:
        gr.Markdown(
            "_Compare needs two artifacts. Pick a Prediction and a Baseline._"
        )
        return
    a_label, b_label = models[0], models[1]
    if a_label == b_label:
        gr.Markdown(
            f"_Compare needs two different artifacts; both sides are `{escape(a_label)}`._"
        )
        return

    pred = _build_compare_view(store, file_id, a_label, kind="prediction")
    base = _build_compare_view(store, file_id, b_label, kind="baseline")
    if pred is None or base is None:
        gr.Markdown(
            f"_Cannot build comparison views for "
            f"`{escape(a_label)}` vs `{escape(b_label)}` on `{escape(file_id)}`._"
        )
        return

    # Compare mode shows only the text-comparison block — the document
    # and layout blocks aren't useful here and add noise.
    try:
        cmp_cls = _registry.get("comparisons", "text")
        renderer_cls = _registry.get("comparison_renderers", "text")
    except Exception:  # noqa: BLE001
        gr.Markdown("_Text comparison is not registered._")
        return
    try:
        result = cmp_cls().compare(pred, base)
    except Exception as e:  # noqa: BLE001
        gr.Markdown(f"_text comparison failed: `{escape(str(e))}`_")
        return
    if result is None:
        gr.Markdown(
            f"_No text comparison for `{escape(a_label)}` vs "
            f"`{escape(b_label)}` — both sides need text._"
        )
        return
    fragment = renderer_cls().render_gradio(
        result, prediction_label=a_label, baseline_label=b_label,
    )
    gr.HTML(fragment)


def _build_compare_view(
    store: ViewerStore, file_id: str, label: str, *, kind: str
) -> PredictionView | BaselineView | None:
    if label == REFERENCE_PSEUDO_MODEL:
        baselines = store.baselines_for(file_id)
        if not baselines:
            return None
        bl = baselines[0]
        if kind == "baseline":
            return BaselineView(
                page_id=file_id,
                label=bl.label,
                text=bl.text,
                provenance=bl.provenance,
            )
        return PredictionView(page_id=file_id, label=bl.label, text=bl.text)
    row = store.get(file_id, label)
    if row is None:
        return None
    document = _document_from_row(row)
    text = row.text or row.markdown
    if kind == "baseline":
        return BaselineView(
            page_id=file_id, label=label, text=text, document=document,
        )
    return PredictionView(
        page_id=file_id, label=label, text=text, document=document,
    )


def _document_from_row(row: ModelRow) -> Any:
    if not row.document_json:
        return None
    try:
        from docling_core.types.doc.document import DoclingDocument
    except ImportError:
        return None
    try:
        return DoclingDocument.model_validate_json(row.document_json)
    except Exception:  # noqa: BLE001
        return None


def _emit_text_body(
    row: ModelRow, store: ViewerStore, *, use_structured: bool
) -> None:
    if not use_structured or not row.items:
        gr.Markdown(
            row.markdown or "_(empty)_",
            elem_classes="ocrscout-markdown-pane",
        )
        return
    parts = ['<div class="ocrscout-markdown-pane ocrscout-structured">']
    for item in row.items:
        color = store.LABEL_COLORS.get(item.label, "#888")
        body_html = item.html or escape(item.text or "").replace("\n", "<br>")
        parts.append(
            f'<div class="ocrscout-section" '
            f'data-label="{escape(item.label)}" '
            f'style="--section-color:{color}">'
            f'<span class="ocrscout-section-tag">{escape(item.label)}</span>'
            f'<div class="ocrscout-section-text">{body_html}</div>'
            f"</div>"
        )
    parts.append("</div>")
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


def _emit_baseline_stats_strip(baseline: BaselineRow) -> None:
    chars = len(baseline.text)
    prov = baseline.provenance
    prov_bits: list[str] = []
    if prov is not None:
        prov_bits.append(prov.method)
        if prov.engine:
            prov_bits.append(prov.engine)
        if prov.confidence is not None:
            prov_bits.append(f"conf={prov.confidence:.2f}")
    prov_html = (
        f'<span class="prov">{escape(", ".join(prov_bits))}</span>'
        if prov_bits else ""
    )
    gr.HTML(
        '<div class="ocrscout-stats-strip ocrscout-baseline-strip">'
        f'<span class="name">{escape(baseline.label)}</span>'
        '<span class="kind">(reference)</span>'
        f"<span>chars: {chars}</span>"
        f"{prov_html}"
        "</div>"
    )


def _emit_baseline_text_body(baseline: BaselineRow) -> None:
    gr.Markdown(
        baseline.text or "_(empty)_",
        elem_classes=["ocrscout-markdown-pane", "ocrscout-baseline-pane"],
    )


_HELP_HTML = """
<div class="ocrscout-help">
<h3>Keyboard shortcuts</h3>
<dl>
  <dt>j / k</dt><dd>Next / previous page</dd>
  <dt>1 / 2 / 3</dt><dd>Switch to Single / Side-by-side / Compare mode</dd>
  <dt>i</dt><dd>Toggle the image pane</dd>
  <dt>?</dt><dd>Show this help</dd>
</dl>
<h3>View modes</h3>
<dl>
  <dt>Single</dt><dd>One model, full markdown — for reading.</dd>
  <dt>Side-by-side</dt><dd>One column per selected model. All models
    selected by default; uncheck to narrow. The grid wraps responsively
    so 8 panes flow into multiple rows on narrow viewports.</dd>
  <dt>Compare</dt><dd>Side-by-side text diff between any two artifacts.
    Baseline can be a model or the page's <code>reference</code> baseline
    (with provenance shown).</dd>
</dl>
<h3>URL parameters</h3>
<p>Append <code>?file=...&amp;models=a,b&amp;mode=Compare</code> to share a
specific view.</p>
</div>
"""
