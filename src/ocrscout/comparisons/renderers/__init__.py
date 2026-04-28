"""Built-in renderers for the standard comparison results.

One module per ``ComparisonResult`` subclass. Every renderer implements the
``ComparisonRenderer`` ABC's three surfaces (``render_html`` for inspect's
one-shot HTTP page, ``render_terminal`` for the Rich console, and
``render_gradio`` for embedding inside the viewer's Compare mode).
"""

from __future__ import annotations
