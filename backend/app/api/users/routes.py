from flask import Blueprint, request, jsonify, make_response, g, current_app
from datetime import datetime, timedelta
from werkzeug.security import check_password_hash, generate_password_hash
import jwt
from flask_restx import Resource, Namespace, fields
import os

from .models import User
from app.api.videos.models import Video
from app.api.videos.routes import video_info
from app.common.decorators import token_required
from app.api.shared_models import empty_input_model, message_model
from app.common.messages import success_msg, error_msg
from app.config import RESOURCE_PATH_URL

api = Namespace("Users", description="Users auth and profile")

login = api.model(
    "Login",
    {
        "email": fields.String(required=True, description="User email"),
        "password": fields.String(required=True, description="User password"),
    },
)

jwt_token = api.model(
    "JWTToken",
    {
        "accessToken": fields.String(required=True, description="JWT token"),
    },
)


@api.route("/login")
class Login(Resource):
    @api.response(201, "Successfully logged in", jwt_token)
    @api.response(404, "User not found", message_model)
    @api.response(403, "Wrong password", message_model)
    @api.expect(login)
    def post(self):
        """
        Authenticate a user and return a JWT token.
        """
        auth = request.json
        user = g.session.query(User).filter(User.email == auth["email"]).first()
        if not user:
            return error_msg("User not found"), 404
        if not check_password_hash(user.password, auth["password"]):
            return error_msg("Wrong password"), 403
        token = jwt.encode(
            {
                "exp": datetime.utcnow() + timedelta(hours=12),
                "id": user.id,
            },
            current_app.config["SECRET_KEY"],
        )
        return {"accessToken": token}, 201


register = api.model(
    "Register",
    {
        "email": fields.String(required=True, description="User email"),
        "password": fields.String(required=True, description="User password"),
        "trueName": fields.String(required=True, description="User true name"),
        "department": fields.String(required=True, description="User department"),
        "bio": fields.String(required=True, description="User biography"),
    },
)


@api.route("/register")
class Register(Resource):
    @api.expect(register)
    @api.response(201, "Successfully registered", message_model)
    @api.response(409, "User already exists", message_model)
    def post(self):
        """
        Register a new user.
        """
        data = request.json
        user = g.session.query(User).filter(User.email == data["email"]).first()
        if user:
            return error_msg("User already exists"), 409
        user = User(
            email=data["email"],
            password=generate_password_hash(data["password"]),
            true_name=data["trueName"],
            department=data["department"],
            bio=data["bio"],
        )
        g.session.add(user)
        g.session.commit()
        return success_msg("successfully registered"), 201


profile_info = api.model(
    "ProfileInfo",
    {
        "id": fields.Integer(required=True, description="User id"),
        "priority": fields.Integer(required=True, description="User priority"),
        "email": fields.String(required=True, description="User email"),
        "trueName": fields.String(required=True, description="User true name"),
        "department": fields.String(required=True, description="User department"),
        "bio": fields.String(required=True, description="User biography"),
        "videos": fields.List(
            fields.Nested(video_info), required=True, description="User videos"
        ),
    },
)


@api.route("/profile")
class Profile(Resource):
    @api.expect(empty_input_model, validate=False)
    @api.marshal_with(profile_info)
    @token_required
    def get(self, current_user: User):
        """
        Get user profile.
        """
        videos = g.session.query(Video).filter(Video.user_id == current_user.id)
        videos = [
            {
                "key": v.id,
                "id": v.id,
                "name": v.original_name,
                "analyzed": v.analyzed,
                "frameCnt": v.frame_cnt,
                "fps": v.fps,
                "width": v.width,
                "height": v.height,
                "timeUploaded": v.time_uploaded.isoformat(),
                "size": v.size,
                "url": f"{RESOURCE_PATH_URL}/videos/{current_user.id}/{v.name}",
            }
            for v in videos
        ]
        return {
            "id": current_user.id,
            "priority": current_user.priority,
            "email": current_user.email,
            "trueName": current_user.true_name,
            "department": current_user.department,
            "bio": current_user.bio,
            "videos": videos,
        }
