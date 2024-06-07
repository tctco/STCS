import React, { useEffect, useState } from "react";
import { Empty, Pagination, Image, Space, Typography } from "antd";
import { useParams } from "react-router-dom";
import userService from "../services/user.service";
import { logError } from "../utils";

const { Title } = Typography;

const ModelDetailPage: React.FC = () => {
  const { modelId } = useParams();
  useEffect(() => {
    userService
      .getModelById(modelId)
      .then((res) => {
        console.log(res.data);
        setImageUrls(res.data.visImages);
      })
      .catch(logError);
  }, []);
  const [imageUrls, setImageUrls] = useState<string[]>([]);
  const [currentImgIdx, setCurrentImageIdx] = useState(0);
  return imageUrls.length > 0 ? (
    <Space direction="vertical" style={{ textAlign: "center" }}>
      <Title level={3}>{imageUrls[currentImgIdx - 1].split("/").pop()}</Title>
      <Image
        height={400}
        src={"http://localhost/" + imageUrls[currentImgIdx - 1]}
      />
      <Pagination
        defaultCurrent={currentImgIdx + 1}
        total={imageUrls.length}
        defaultPageSize={1}
        onChange={setCurrentImageIdx}
      />
    </Space>
  ) : (
    <Empty description="Currently there is no training image to be displayed" />
  );
};

export default ModelDetailPage;
