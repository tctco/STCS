import axios from "axios";
axios.defaults.baseURL =
  import.meta.env.VITE_APP_API_URL || "http://localhost:5000/api";
axios.defaults.headers.common = {
  "Content-Type": "application/json",
};

import ReactDOM from "react-dom/client";
import "./index.css";
import { HashRouter, Route, Routes } from "react-router-dom";
import LoginPage from "./pages/login.page";
import ProfilePage from "./pages/profile.page";
import WelcomePage from "./pages/welcome.page";
import RegisterPage from "./pages/register.page";
import DatasetPage from "./pages/dataset.page";

import { Provider } from "react-redux";
import store from "./store";
import AuthRoute from "./components/AuthRoute";
import { useEffect } from "react";
import { Alert } from "antd";

import authService from "./services/auth.service";
import useBrowserCheck from "./hooks/useBrowserCheck";
import Layout, { Route as RouteInterface } from "./components/Layout";
import AnnotationPage from "./pages/annotation.page";
import Breadcrumbs from "./components/BreadCrumbs";
import { DatasetProvider } from "./context/datasetProvider";
import ModelPage from "./pages/model.page";
import ModelDetailPage from "./pages/modelDetail.page";

const route: RouteInterface = {
  routes: [
    { path: "/", name: "Welcome", icon: "smile", element: <WelcomePage /> },
    {
      path: "/Profile",
      name: "Tracking",
      icon: "track",
      element: <ProfilePage />,
    },
    {
      path: "/Datasets",
      name: "Datasets",
      icon: "database",
      element: <DatasetPage />,
    },
    {
      path: "/Models",
      name: "Models",
      icon: "model",
      element: <ModelPage />,
    },
  ],
};

const App = () => {
  const isChrome = useBrowserCheck();

  useEffect(() => {
    authService.checkExpiration();
  }, []);

  return (
    <Provider store={store}>
      <HashRouter>
        <Layout route={route}>
          <Breadcrumbs />
          <DatasetProvider>
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
              <Route
                path="/datasets"
                element={
                  <AuthRoute>
                    <DatasetPage />
                  </AuthRoute>
                }
              />
              <Route
                path="/datasets/:datasetId"
                element={
                  <AuthRoute>
                    <AnnotationPage />
                  </AuthRoute>
                }
              />
              <Route
                path="/models"
                element={
                  <AuthRoute>
                    <ModelPage />
                  </AuthRoute>
                }
              />
              <Route
                path="/models/:modelId"
                element={
                  <AuthRoute>
                    <ModelDetailPage />
                  </AuthRoute>
                }
              />
            </Routes>
          </DatasetProvider>
        </Layout>
        {!isChrome && (
          <Alert
            message="This app is only supported on Chrome"
            type="warning"
            showIcon
            closable
          />
        )}
      </HashRouter>
    </Provider>
  );
};

ReactDOM.createRoot(document.getElementById("root")!).render(
  // <React.StrictMode>
  <App />
  // </React.StrictMode>
);
