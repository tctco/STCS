import {
  LoginOutlined,
  UserAddOutlined,
  LogoutOutlined,
} from "@ant-design/icons";
import { Layout, Button } from "antd";
import { Link, useNavigate } from "react-router-dom";
import { useSelector } from "react-redux";
import { logout } from "../slices/auth";
import { useDispatch } from "react-redux";

const NavBar = () => {
  const isLoggedIn = useSelector((state) => state.auth.isLoggedIn);
  const navigate = useNavigate();
  const dispatch = useDispatch();

  return (
    <>
      {/* <Layout>
        <Layout.Header
          style={{
            display: "flex",
            justifyContent: "space-between",
            background: "#f8f9fa",
          }}
        > */}
      {isLoggedIn ? (
        <div>
          <Button
            type="text"
            danger
            style={{ marginRight: "10px" }}
            icon={<LogoutOutlined />}
            onClick={() => {
              dispatch(logout())
                .unwrap()
                .then(() => navigate("/"));
            }}
          >
            Log out
          </Button>
        </div>
      ) : (
        <div>
          <Link to="/login">
            <Button
              type="text"
              style={{ marginRight: "10px" }}
              icon={<LoginOutlined />}
            >
              Log in
            </Button>
          </Link>
          <Link to="/register">
            <Button icon={<UserAddOutlined />} type="text">
              Register
            </Button>
          </Link>
        </div>
      )}
      {/* </Layout.Header>
      </Layout> */}
    </>
  );
};

export default NavBar;
