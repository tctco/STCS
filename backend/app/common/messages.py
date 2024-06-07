def error_msg(msg: str) -> dict:
    return {"success": False, "message": msg}


def success_msg(msg: str) -> dict:
    return {"success": True, "message": msg}
