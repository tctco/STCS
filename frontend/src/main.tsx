import ReactDOM from "react-dom/client";
import "./index.css";
import { HashRouter, Route, Routes } from "react-router-dom";
import LoginPage from "./pages/login.page";
import ProfilePage from "./pages/profile.page";
import WelcomePage from "./pages/welcome.page";
import RegisterPage from "./pages/register.page";
import NavBar from "./components/NavBar";
import axios from "axios";
import { Provider } from "react-redux";
import store from "./store";
import AuthRoute from "./components/AuthRoute";
import { useEffect, useState } from "react";
import { Alert } from "antd";

axios.defaults.baseURL = import.meta.env.VITE_APP_API_URL || "http://localhost:5000/api";
axios.defaults.headers.common = {
  "Content-Type": "application/json",
};

const App = () => {
  const [isChrome, setIsChrome] = useState(true)

  useEffect(() => {
    if (navigator.userAgent.indexOf("Chrome") === -1) {
      setIsChrome(false)
    }
  }, [])

  return (<Provider store={store}>
    <HashRouter>
      <NavBar />
      {!isChrome && <Alert message="This app is only supported on Chrome" type="warning" showIcon closable />}
      <Routes>
        <Route path="/" element={<WelcomePage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />

        <Route
          path="/profile"
          element={
            <AuthRoute>
              <ProfilePage />
            </AuthRoute>
          }
        />
      </Routes>
    </HashRouter>
  </Provider>)
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  // <React.StrictMode>
  <App />
  // </React.StrictMode>
);
