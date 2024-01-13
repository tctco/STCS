import { useSelector } from "react-redux";
import { Navigate } from "react-router-dom";
import { message } from "antd";

const AuthRoute = ({ children }) => {
  const { isLoggedIn } = useSelector((state) => state.auth);
  console.log(children);
  if (!isLoggedIn) {
    message.error("You must be logged in to view this page.");
  }
  return isLoggedIn ? <>{children}</> : <Navigate to="/login" replace />;
};

export default AuthRoute;
