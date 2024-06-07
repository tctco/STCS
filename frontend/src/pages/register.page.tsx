import { Button, Form, Input, Row, Col, Card, message } from "antd";
// import authService from "../services/auth.service";
import { Link, useNavigate } from "react-router-dom";
import { register, RegisterParams } from "../slices/auth";
import { useEffect, useState } from "react";
import { clearMessage } from "../slices/message";
import { useAppDispatch } from "../hooks/useAppDispatch";

const RegisterPage = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  const [_loading, setLoading] = useState(false);
  useEffect(() => {
    dispatch(clearMessage);
  });

  const onFinish = (values: RegisterParams) => {
    // authService
    //   .register(
    //     values.email as string,
    //     values.password as string,
    //     values.trueName as string,
    //     values.department as string,
    //     values.bio as string
    //   )
    //   .then(() => {
    //     message.success("register success");
    //     navigate("/login", { replace: true });
    //   })
    //   .catch((e) => {
    //     message.error(`Register failed: ${e}`);
    //   });
    setLoading(true);
    dispatch(register(values))
      .unwrap()
      .then(() => navigate("/profile", { replace: true }))
      .catch((e) => {
        console.log(e);
        message.error(`Register failed: ${e.message}`);
        setLoading(false);
      });
  };

  const onFinishFailed = (errorInfo: unknown) => {
    console.log("Failed:", errorInfo);
  };
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
          <Card title="Register" style={{ width: 400 }}>
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
              <Form.Item<RegisterParams>
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

              <Form.Item<RegisterParams>
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
              <Form.Item<RegisterParams>
                label="True name"
                name="trueName"
                rules={[
                  {
                    required: true,
                    message:
                      "You are required to provide your true name to use our app",
                  },
                ]}
              >
                <Input />
              </Form.Item>
              <Form.Item<RegisterParams>
                label="Bio"
                name="bio"
                rules={[
                  {
                    required: true,
                    message:
                      "Please introduce yourself and demonstrate how you will use our app",
                  },
                ]}
              >
                <Input />
              </Form.Item>
              <Form.Item<RegisterParams>
                label="Department"
                name="department"
                rules={[
                  {
                    required: true,
                    message:
                      "Please fill in which department you are working for",
                  },
                ]}
              >
                <Input />
              </Form.Item>

              <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
                <Button type="primary" htmlType="submit">
                  Submit
                </Button>{" "}
                Or <Link to="/login">login now!</Link>
              </Form.Item>
            </Form>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default RegisterPage;
