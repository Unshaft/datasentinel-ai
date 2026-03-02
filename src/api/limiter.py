"""
Limiteur de taux partagé pour tous les routers FastAPI.

Usage dans un router :
    from src.api.limiter import limiter
    from fastapi import Request

    @router.post("")
    @limiter.limit("20/minute")
    async def my_endpoint(request: Request, ...):
        ...

Puis dans main.py :
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
