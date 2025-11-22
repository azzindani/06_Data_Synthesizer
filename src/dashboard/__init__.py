"""Dashboard module for monitoring."""

def __getattr__(name):
    if name == 'DashboardServer':
        from .app import DashboardServer
        return DashboardServer
    if name == 'create_dashboard':
        from .app import create_dashboard
        return create_dashboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['DashboardServer', 'create_dashboard']
