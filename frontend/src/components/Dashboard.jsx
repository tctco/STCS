import React from "react";
import { Layout, Menu, PageHeader } from "antd";
import PreprocessPage from "../PreprocessPage/PreprocessPage";
// import VisibleSummaryPage from "../SummaryPage/SummaryPage";
// import VisibleMonitorPage from "../../Containers/VisibleMonitorPage";
// import {AnnotatorAndTrainer} from "../../Utils/Annotator";
import {
  PieChartOutlined,
  DesktopOutlined,
  FileOutlined,
  SmileOutlined,
  createFromIconfontCN,
  EditOutlined,
} from "@ant-design/icons";
import styles from "./Dashboard.module.css";
import AppFooter from "../AppFooter/AppFooter";

const IconFont = createFromIconfontCN({
  scriptUrl: "////at.alicdn.com/t/font_2723840_px795wck4bq.js",
});

const { Footer, Sider, Content } = Layout;

export default class Dashboard extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      collapsed: false,
      selected: "Welcome",
      selectedPath: ["Welcome"],
    };
    this.handleMenuClick = this.handleMenuClick.bind(this);
    this.keyContentMap = {
      // Summary: <VisibleSummaryPage />,
      Welcome: <PreprocessPage />,
      // Monitor: <VisibleMonitorPage />,
      // Annotator: <AnnotatorAndTrainer />,
    };
  }

  onCollapse = (collapsed) => {
    this.setState({ collapsed });
  };

  handleMenuClick({ item, key, keyPath, event }) {
    this.setState({
      selected: key,
      selectedPath: keyPath,
    });
    this.props.setMenu(key);
  }

  render() {
    return (
      <>
        <Layout style={{ minHeight: "100vh" }}>
          <Sider
            collapsible
            collapsed={this.state.collapsed}
            onCollapse={this.onCollapse}
            breakpoint="lg"
            collapsedWidth="50px"
          >
            <div className={styles.logo}>
              <IconFont type="icon-huabanfuben" style={{ fontSize: 44 }} />
              <div
                style={{ display: this.state.collapsed ? "none" : "inline" }}
              >
                BehavLab
              </div>
            </div>
            <Menu
              theme="dark"
              defaultSelectedKeys={["Welcome"]}
              mode="inline"
              selectedKeys={[this.props.menu]}
              onClick={this.handleMenuClick}
            >
              <Menu.Item key="Welcome" icon={<SmileOutlined />}>
                Welcome
              </Menu.Item>
              <Menu.Item key="Summary" icon={<PieChartOutlined />}>
                Summary
              </Menu.Item>
              <Menu.Item key="Monitor" icon={<DesktopOutlined />}>
                Monitor
              </Menu.Item>
              <Menu.Item key="Documentation" icon={<FileOutlined />}>
                Documentation
              </Menu.Item>
              <Menu.Item key="Annotator" icon={<EditOutlined />}>
                Annotator
              </Menu.Item>
            </Menu>
          </Sider>
          <Layout>
            <PageHeader title={this.props.menu} />
            <Content style={{ backgroundColor: "#e8e8e8" }}>
              <div style={{ padding: 12 }}>
                {this.keyContentMap[this.props.menu]}
              </div>
            </Content>
            <Footer>
              <AppFooter />
            </Footer>
          </Layout>
        </Layout>
      </>
    );
  }
}
