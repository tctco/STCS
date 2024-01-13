import userService from "../services/user.service";
import {
  message,
  Checkbox,
  Upload,
  Table,
  Button,
  Tag,
  Space,
  Modal,
  InputNumber,
  Form,
  Tooltip,
  List,
  Avatar,
  Descriptions,
  Empty,
  Popconfirm,
  Select,
} from "antd";
import {
  UploadOutlined,
  SendOutlined,
  DeleteOutlined,
} from "@ant-design/icons";
import { useState, useEffect } from "react";
import { User, Video, Task, Model } from "../types";
import type { ColumnsType } from "antd/es/table";
import type { UploadProps } from "antd";
import axios from "axios";
import authHeader from "../services/auth-header";
import { VideoWithSVG } from "../components/VideoPlayer";
import TabContainer from "../components/TabContainer";
import Resizer from "../components/Resizer";
import { randomColor } from "../utils";
import VerticalSplitPanel from "../components/VerticalSplitPanel.jsx";

function getStatusColor(status: string) {
  switch (status.toLowerCase()) {
    case "queued":
      return "#FFCC00"; // Yellow
    case "started":
      return "#007BFF"; // Blue
    case "deferred":
      return "#FFA500"; // Orange
    case "finished":
      return "#28A745"; // Green
    case "stopped":
      return "#6C757D"; // Gray
    case "scheduled":
      return "#17A2B8"; // Teal
    case "failed":
      return "#DC3545"; // Red
    case "canceled":
      return "#343A40"; // Dark Gray
    default:
      return "#000"; // Black for unknown status
  }
}

const columns: ColumnsType<Video> = [
  {
    title: "Name",
    dataIndex: "name",
    key: "name",
  },
  {
    title: "Frames",
    dataIndex: "frameCnt",
    key: "frameCnt",
  },
  {
    title: "FPS",
    dataIndex: "fps",
    key: "fps",
    render: (value: number) => value.toFixed(2).toString(),
  },
  {
    title: "Resolution",
    key: "resolution",
    render: (_, record) => (
      <>
        {record.width}x{record.height}
      </>
    ),
  },
  {
    title: "Upload Time",
    dataIndex: "timeUploaded",
    key: "timeUploaded",
    sorter: (a, b) => (a.timeUploaded > b.timeUploaded ? 1 : -1),
    sortDirections: ["descend", "ascend"],
  },
  {
    title: "Analyzed",
    dataIndex: "analyzed",
    key: "analyzed",
    render: (value) => (
      <Tag color={value ? "green" : "red"}>
        {value ? "Analyzed" : "Not Analyzed"}
      </Tag>
    ),
  },
  {
    title: "Size (MB)",
    dataIndex: "size",
    render: (value) => `${(value / 1024 / 1024).toFixed(2)}`,
  },
  {
    title: "Action",
    key: "action",
    render: (_, record) => (
      <Popconfirm
        title="Delete the video"
        description="Are you sure you want to delete this video?"
        okText="Yes"
        cancelText="No"
        onConfirm={() =>
          userService
            .deleteVideo(record.id)
            .then((res) => message.success(res.data.message))
            .catch((e) => message.error(e.message))
        }
      >
        <Button type="link" danger icon={<DeleteOutlined />}>
          Delete
        </Button>
      </Popconfirm>
    ),
  },
];

const ProfilePage = () => {
  const [user, setUser] = useState<User>(null);
  const [selectedVideo, setSelectedVideo] = useState<Video>(null);
  const [isConfirmOpen, setConfirmOpen] = useState<boolean>(false);
  const [maxDet, setMaxDet] = useState<number>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [enableFlow, setEnableFlow] = useState<boolean>(true);
  const [animalClass, setAnimalClass] = useState<string>(null);
  const [trainedModels, setTrainedModels] = useState<Model[]>([])
  const [segmModelID, setSegmModelID] = useState<string>(null)
  const [poseModelID, setPoseModelID] = useState<string>(null)
  const [flowModelID, setFlowModelID] = useState<string>(null)
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout>(null) // For polling jobs

  const uploadProps: UploadProps = {
    name: "video",
    action: axios.defaults.baseURL + "/videos/",
    method: "PUT",
    headers: authHeader(),
    maxCount: 1,
    beforeUpload(file) {
      const isMP4 = file.type === "video/mp4";
      if (!isMP4) {
        message.error(
          `${file.name} is not a mp4 file. Consider converting it to MP4 file with H264 codec.`
        );
      }
      const isLt2G = file.size / 1024 / 1024 / 1024 < 2;
      if (!isLt2G) {
        message.error("Video must smaller than 2GB!");
      }
      return isMP4 && isLt2G;
    },
  };

  useEffect(() => {
    setSegmModelID(null)
    setPoseModelID(null)
    setFlowModelID(null)
  }, [animalClass])

  const requestJobs = () => {
    userService
      .getJobs()
      .then((res) => {
        console.log(res.data);
        setTasks(res.data as Task[]);
        setTimeoutId(setTimeout(requestJobs, 1000 * 10));
      })
      .catch((e) => message.error(e.message));
  };

  const rowSelection = {
    onChange: (selectedRowKeys: React.Key[], selectedRows: Video[]) => {
      console.log(
        `selectedRowKeys: ${selectedRowKeys}`,
        "selectedRows: ",
        selectedRows
      );
      setSelectedVideo(selectedRows[0]);
    },
    getCheckboxProps: (record: Video) => ({
      name: record.name,
    }),
  };

  const onChange = (info) => {
    if (info.file.status !== "uploading") {
      console.log(info.file, info.fileList);
    }
    if (info.file.status === "done") {
      message.success(`${info.file.name} file uploaded successfully`);
      userService
        .getProfile()
        .then((res) => {
          console.log(res.data);
          setUser(res.data as User);
        })
        .catch((e) => message.error(e.message));
    } else if (info.file.status === "error") {
      console.log(info);
      message.error(`[Server]: ${info.file.response.message}`)
    }
  };

  useEffect(() => {
    userService
      .getProfile()
      .then((res) => {
        console.log(res.data);
        setUser(res.data as User);
      })
      .catch((e) => message.error(e.message));
    let timeoutId = null;
    
    requestJobs();
    return () => {
      clearTimeout(timeoutId);
    };
  }, []);

  useEffect(() => {
    userService.getModels().then(res => {
      setTrainedModels(res.data as Model[])
      console.log(res.data)
    }).catch(e => message.error(e.message))
  }, [isConfirmOpen])

  const handleConfirm = () => {
    if (typeof maxDet !== "number" || maxDet < 0) {
      message.error("Invalid number of animals");
      return;
    }
    if (animalClass === null) {
      message.error("Empty animal class!");
      return;
    }
    if (segmModelID === null) {
      message.error("Empty instance segmentation model!");
      return;
    }
    if (poseModelID === null) {
      message.error("Empty pose estimation model!");
      return;
    }
    if (flowModelID === null && enableFlow) {
      message.error("Empty flow estimation model!");
      return;
    }
    userService
      .newJob(selectedVideo.id, maxDet, enableFlow, animalClass, segmModelID, poseModelID, flowModelID)
      .then((res) => {
        message.success(res.data.message);
        clearTimeout(timeoutId)
        requestJobs();
      })
      .catch((e) => {
        console.log(e);
        const msg = e.response.data.message
          ? e.response.data.message
          : e.message;
        message.error(msg);
      });
    setConfirmOpen(false);
  };

  return user ? (
    <>
      <TabContainer title="Scheduled Jobs">
        <List
          grid={{ gutter: 16, column: 4 }}
          itemLayout="horizontal"
          dataSource={tasks}
          style={{ maxHeight: 200, overflow: "auto", padding: 10 }}
          locale={{ emptyText: "No scheduled jobs yet" }}
          renderItem={(item: Task) => (
            <List.Item>
              <List.Item.Meta
                avatar={
                  <Avatar
                    style={{ backgroundColor: randomColor(item.trueName) }}
                  >
                    {item.trueName[0]}
                  </Avatar>
                }
                title={
                  <Space>
                    <span style={{ color: getStatusColor(item.status) }}>
                      {item.status.toUpperCase()}
                    </span>
                    {item.owned &&
                    (item.status.toLowerCase() === "started" ||
                      item.status.toLocaleLowerCase() === "queued") ? (
                      <Button
                        danger
                        onClick={() => {
                          userService
                            .cancelJob(item.taskId)
                            .then((res) => {
                              message.success(res.data.message)
                              clearTimeout(timeoutId)
                              requestJobs()
                            })
                            .catch((e) => {
                              message.error(e.message);
                            });
                        }}
                      >
                        Cancel
                      </Button>
                    ) : (
                      <></>
                    )}
                  </Space>
                }
                description={
                  <Descriptions column={2}>
                    <Descriptions.Item label="Animal(s)">
                      {item.maxDet}
                    </Descriptions.Item>
                    <Descriptions.Item label="Priority">
                      {item.priority === 0 ? "Low" : "High"}
                    </Descriptions.Item>
                    <Descriptions.Item label="Created" span={2}>
                      {item.created}
                    </Descriptions.Item>
                    <Descriptions.Item label="Started" span={2}>
                      {item.started ? item.started : "Not yet"}
                    </Descriptions.Item>
                  </Descriptions>
                }
              />
            </List.Item>
          )}
        />
      </TabContainer>

      {selectedVideo ? (
        <Modal
          open={isConfirmOpen && selectedVideo !== null}
          onOk={handleConfirm}
          onCancel={() => setConfirmOpen(false)}
          title={selectedVideo.name}
        >
          <Form>
            <Form.Item
              label="Maximum # of animals"
              rules={[
                {
                  required: true,
                  message: "Empty or invalid number of animals",
                },
              ]}
            >
              <InputNumber min={1} onChange={(v) => setMaxDet(v)}></InputNumber>
            </Form.Item>
            <Form.Item label="Enable optical flow?">
              <Tooltip
                placement="bottom"
                title="Optical flow generally makes tracking better, but requires more time to track"
              >
                <Checkbox
                  onChange={(e) => setEnableFlow(e.target.checked)}
                  checked={enableFlow}
                />
              </Tooltip>
            </Form.Item>
            <Form.Item
              label="Animal"
              rules={[{ required: true, message: "Empty animal" }]}
            >
              <Select
                value={animalClass}
                onChange={setAnimalClass}
                options={[
                  { value: "mouse", label: "Mouse" },
                  { value: "ant", label: "Ant" },
                  {value: "fly", label: "Fly"}
                ]}
              />
            </Form.Item>
            <Form.Item
              label="Segm Model"
              rules={[{ required: true, message: "Empty instance segmentation model!" }]}
            >
              <Select
                value={segmModelID}
                onChange={setSegmModelID}
                options={trainedModels.filter(m => m.type === "segm" && m.animal === animalClass).map(m => ({value: m.id, label: m.id+` (${m.method})`}))}
              />
            </Form.Item>
            <Form.Item
              label="Pose Model"
              rules={[{ required: true, message: "Empty pose estimation model!" }]}
            >
              <Select
                value={poseModelID}
                onChange={setPoseModelID}
                options={trainedModels.filter(m => m.type === "pose" && m.animal === animalClass).map(m => ({value: m.id, label: m.id+` (${m.method})`}))}
              />
            </Form.Item>
            <Form.Item
              label="Flow Model"
              rules={[{ required: true, message: "Empty flow estimation model!" }]}
            >
              <Select
                value={flowModelID}
                onChange={setFlowModelID}
                disabled={!enableFlow}
                options={trainedModels.filter(m => m.type === "flow").map(m => ({value: m.id, label: m.id+` (${m.method})`}))}
              />
            </Form.Item>
          </Form>
        </Modal>
      ) : (
        <></>
      )}
      <VerticalSplitPanel
        left={
          <Resizer>
            <TabContainer title="Uploaded Videos">
              <div style={{ margin: 10 }}>
                <Table
                  columns={columns}
                  dataSource={user.videos}
                  scroll={{ x: 500 }}
                  rowSelection={{ ...rowSelection, type: "radio" }}
                  size="small"
                  footer={() => (
                    <div style={{ textAlign: "center" }}>
                      <Space>
                        <Upload {...uploadProps} onChange={onChange}>
                          <Button icon={<UploadOutlined />}>
                            Click to Upload
                          </Button>
                        </Upload>
                        <Button
                          disabled={selectedVideo === null}
                          icon={<SendOutlined />}
                          onClick={() => setConfirmOpen(true)}
                        >
                          Start Tracking
                        </Button>
                      </Space>
                    </div>
                  )}
                />
              </div>
            </TabContainer>
          </Resizer>
        }
        right={
          <Resizer>
            <TabContainer title="Viewer">
              {selectedVideo ? (
                <>
                  <VideoWithSVG
                    videoId={selectedVideo.id}
                    src={selectedVideo.url}
                    fps={selectedVideo.fps}
                    width={selectedVideo.width}
                    height={selectedVideo.height}
                  />
                </>
              ) : (
                <Empty
                  imageStyle={{ height: 250 }}
                  description="Select an uploaded video to start"
                />
              )}
            </TabContainer>
          </Resizer>
        }
      />
    </>
  ) : (
    <></>
  );
};

export default ProfilePage;
