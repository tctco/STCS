import React, { useState } from "react";
import BasicLayout, { MenuDataItem } from "@ant-design/pro-layout";
import NavBar from "./NavBar";
import { useLocation } from "react-router-dom";
import { Link } from "react-router-dom";
import {
  HomeOutlined,
  SmileOutlined,
  AlignLeftOutlined,
  DatabaseOutlined,
  RadarChartOutlined,
} from "@ant-design/icons";

const IconMap = {
  home: <HomeOutlined />,
  smile: <SmileOutlined />,
  track: <AlignLeftOutlined />,
  database: <DatabaseOutlined />,
  model: <RadarChartOutlined />,
};

export interface Route {
  path?: string;
  routes: Array<{
    exact?: boolean;
    icon?: string;
    name?: string;
    path: string;
    // optional secondary menu
    children?: Route["routes"];
    element?: React.ReactNode;
  }>;
}

interface LayoutProps {
  children?: React.ReactNode;
  route: Route;
}

export const Layout: React.FC<LayoutProps> = ({ children, route }) => {
  const location = useLocation();
  const [settings, _setSettings] = useState({
    layout: "mix" as "mix" | "side" | "top",
  });

  const loopMenuItem = (menus: MenuDataItem[]): MenuDataItem[] =>
    menus.map(({ icon, children, ...item }) => ({
      ...item,
      icon: icon && IconMap[icon as string],
      children: children && loopMenuItem(children),
    }));
  return (
    <BasicLayout
      style={{ minHeight: "100vh" }}
      {...settings}
      logo="/logo.svg"
      location={location}
      title="segTracker"
      route={route}
      rightContentRender={() => <NavBar />}
      menuDataRender={() => loopMenuItem(route.routes)}
      menuItemRender={(menuItemProps, defaultDom) => {
        if (menuItemProps.isUrl || !menuItemProps.path) {
          return defaultDom;
        }
        return <Link to={menuItemProps.path}>{defaultDom}</Link>;
      }}
    >
      {children}
    </BasicLayout>
  );
};

export default Layout;
