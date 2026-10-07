"""Application business rules: one use case per thing a caller can ask this service to do.

Use cases orchestrate the domain through its ports. They know nothing about HTTP, FastAPI or Laya;
the composition root (``laya_client.container``) hands each one the adapters it needs.
"""
