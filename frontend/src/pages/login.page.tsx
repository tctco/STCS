import { Button, Checkbox, Form, Input, Row, Col, Card, message } from "antd";
// import authService from "../services/auth.service";
import { Link, useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { login } from "../slices/auth";
import { clearMessage } from "../slices/message";
import { useEffect, useState } from "react";

type FieldType = {
  email?: string;
  password?: string;
  remember?: string;
};

const LoginPage = () => {
  const { isLoggedIn } = useSelector((state: any) => state.auth);
  const [_, setLoading] = useState(false);
  const navigate = useNavigate();
  const dispatch = useDispatch();
  useEffect(() => {
    dispatch(clearMessage);
  }, [dispatch]);
  const onFinish = (values: FieldType) => {
    // authService
    //   .login(values.email as string, values.password as string)
    //   .then(() => {
    //     message.success("Login success");
    //     navigate("/profile", { replace: true });
    //   })
    //   .catch((e) => {
    //     message.error(`Login failed: ${e}`);
    //   });
    setLoading(true);
    dispatch(login(values))
      .unwrap()
      .then(() => {
        navigate("/profile", { replace: true });
      })
      .catch((e) => {
        console.log(e);
        message.error(`Login failed: ${e.message}`);
        setLoading(false);
      });
  };

  const onFinishFailed = (errorInfo: unknown) => {
    console.log("Failed:", errorInfo);
  };
  if (isLoggedIn) {
    navigate("/profile", { replace: true });
  }
  return (
    <div
      style={{
        display: "flex",
        marginTop: 100,
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <Row justify="center">
        <Col>
          <Card title="Login" style={{ width: 400 }}>
            <Form
              name="basic"
              labelCol={{ span: 8 }}
              wrapperCol={{ span: 16 }}
              style={{ maxWidth: 600 }}
              initialValues={{ remember: true }}
              onFinish={onFinish}
              onFinishFailed={onFinishFailed}
              autoComplete="off"
            >
              <Form.Item<FieldType>
                label="email"
                name="email"
                rules={[
                  {
                    type: "email",
                    required: true,
                    message: "Empty or invalid email address",
                  },
                ]}
              >
                <Input />
              </Form.Item>

              <Form.Item<FieldType>
                label="Password"
                name="password"
                rules={[
                  {
                    required: true,
                    message: "Password has to be at least 6 characters long",
                    min: 6,
                  },
                ]}
              >
                <Input.Password />
              </Form.Item>

              <Form.Item<FieldType>
                name="remember"
                valuePropName="checked"
                wrapperCol={{ offset: 8, span: 16 }}
              >
                <Checkbox>Remember me</Checkbox>
              </Form.Item>

              <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
                <Button type="primary" htmlType="submit">
                  Submit
                </Button>{" "}
                Or <Link to="/register">register now!</Link>
              </Form.Item>
            </Form>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default LoginPage;
