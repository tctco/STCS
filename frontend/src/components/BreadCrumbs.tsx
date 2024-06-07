import { useLocation, Link } from "react-router-dom";
import { Breadcrumb } from "antd";

const Breadcrumbs = () => {
  const location = useLocation();
  const pathSnippets = location.pathname.split("/").filter((i) => i);

  const breadcrumbItems = pathSnippets.map((_, index) => {
    const url = `/${pathSnippets.slice(0, index + 1).join("/")}`;
    return (
      <Breadcrumb.Item key={url}>
        <Link to={url}>{url.split("/").pop()}</Link>
      </Breadcrumb.Item>
    );
  });
  return (
    <Breadcrumb style={{ margin: "0 0 24px 0" }}>{breadcrumbItems}</Breadcrumb>
  );
};

export default Breadcrumbs;
