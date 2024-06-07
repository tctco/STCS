from functools import wraps
from flask import current_app, request, g
import jwt
from app.api.users.models import User


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # jwt is passed in the request header
        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[1]
        # return 401 if token is not passed
        if not token:
            return "missing token!", 401

        try:
            # decoding the payload to fetch the stored details
            data = jwt.decode(token, current_app.config["SECRET_KEY"], ["HS256"])
            current_user = g.session.query(User).filter(User.id == data["id"]).first()
        except Exception as e:
            return "invalid token!", 401
        # returns the current logged in users context to the routes
        return f(*args, current_user=current_user, **kwargs)

    return decorated
