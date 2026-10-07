"""Enterprise business rules: entities, errors, policies and the ports the outer layers implement.

Nothing in this package imports a framework, an ORM or the Laya library itself. The dependency rule
points inward: application, infrastructure and interfaces depend on this package, never the reverse.
"""
