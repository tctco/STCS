from flask_restx import Namespace, fields

api = Namespace("SharedModels", description="Shared api models")

empty_input_model = api.model("EmptyInput", {})
message_model = api.model(
    "Message", {"message": fields.String(), "success": fields.Boolean()}
)
