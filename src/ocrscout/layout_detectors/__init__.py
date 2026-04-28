"""Layout detector implementations.

Detectors are registered through :mod:`ocrscout.registry` (group
``layout_detectors``) and consumed by layout-aware backends like
:class:`ocrscout.backends.layout_chat.LayoutChatBackend`.

Importing this package never triggers an ONNX/PyTorch import — concrete
detectors lazy-import their runtimes on first ``detect()`` call.
"""
