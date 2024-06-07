import userService from "../services/user.service";
import type { Key } from "react";
import {
  message,
  Upload,
  Table,
  Button,
  Tag,
  Avatar,
  Descriptions,
  Popconfirm,
  Progress,
  Space,
} from "antd";
import {
  ModalForm,
  ProForm,
  ProFormCheckbox,
  ProFormDependency,
  ProFormDigit,
  ProFormInstance,
  ProFormSelect,
  ProList,
} from "@ant-design/pro-components";
import { UploadChangeParam } from "antd/es/upload/interface.js";
import {
  UploadOutlined,
  SendOutlined,
  DeleteOutlined,
} from "@ant-design/icons";
import React, { useState, useEffect, useRef } from "react";
import { User, Video, Job, Model } from "../types";
import type { ColumnsType } from "antd/es/table";
import type { UploadProps } from "antd";
import authHeader from "../services/auth-header";
import { VideoWithSVG } from "../components/VideoPlayer.jsx";
import TabContainer from "../components/TabContainer";
import Resizer from "../components/Resizer";
import { randomColor } from "../utils";
import VerticalSplitPanel from "../components/VerticalSplitPanel.jsx";
import { VideoProvider, useVideo } from "../context/videoProvider";
import { JobProvider, useJob } from "../context/JobProvider";
import { VideoTracksPlot } from "../charts/TrackletsPlot.js";
import ButtonWithConfirmation from "../components/ConfirmButton.js";

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
    render: (value) => new Date(value).toLocaleString(),
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
            .catch((e) => message.error(e.message, 10))
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

  const onChange = (info: UploadChangeParam) => {
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
        .catch((e) => message.error(e.message, 10));
    } else if (info.file.status === "error") {
      console.log(info);
      message.error(`[Server]: ${info.file.response.message}`, 10);
    }
  };

  useEffect(() => {
    userService
      .getProfile()
      .then((res) => {
        console.log(res.data);
        setUser(res.data as User);
      })
      .catch((e) => message.error(e.message, 10));
  }, []);

  return user ? (
    <VideoProvider>
      <JobProvider>
        <VerticalSplitPanel
          left={
            <>
              <Resizer>
                <TabContainer title="Scheduled Jobs">
                  <JobList />
                </TabContainer>
              </Resizer>
              <Resizer>
                <TabContainer title="Uploaded Videos">
                  <div style={{ margin: 10 }}>
                    <VideoTable videosMeta={user.videos} onChange={onChange} />
                  </div>
                </TabContainer>
              </Resizer>
            </>
          }
          right={
            <>
              <Resizer>
                <TabContainer title="Viewer">
                  <>
                    <VideoWithSVG />
                    <ConfigForm />
                  </>
                </TabContainer>
              </Resizer>
              <TabContainer title="Tracklets">
                <VideoTracksPlot />
              </TabContainer>
            </>
          }
        />
      </JobProvider>
    </VideoProvider>
  ) : (
    <></>
  );
};

export default ProfilePage;

interface VideoTableProps {
  videosMeta: Video[];
  onChange: (info: UploadChangeParam) => void;
}

const VideoTable: React.FC<VideoTableProps> = ({ videosMeta, onChange }) => {
  const { setVideoMeta } = useVideo();
  const uploadProps: UploadProps = {
    name: "video",
    action: userService.getUploadVideoApi(),
    headers: authHeader(),
    maxCount: 1,
    beforeUpload(file) {
      const isMP4 = file.type === "video/mp4";
      if (!isMP4) {
        message.error(
          `${file.name} is not a mp4 file. Consider converting it to MP4 file with H264 codec.`,
          10
        );
      }
      const isLt2G = file.size / 1024 / 1024 / 1024 < 2;
      if (!isLt2G) {
        message.error("Video must smaller than 2GB!", 10);
      }
      return isMP4 && isLt2G;
    },
  };
  const rowSelection = {
    onChange: (selectedRowKeys: React.Key[], selectedRows: Video[]) => {
      console.log(
        `selectedRowKeys: ${selectedRowKeys}`,
        "selectedRows: ",
        selectedRows
      );
      setVideoMeta(selectedRows[0]);
    },
    getCheckboxProps: (record: Video) => ({
      name: record.name,
    }),
  };
  return (
    <Table
      columns={columns}
      dataSource={videosMeta}
      scroll={{ x: 500 }}
      rowSelection={{ ...rowSelection, type: "radio" }}
      size="small"
      footer={() => (
        <div style={{ textAlign: "center" }}>
          <Upload {...uploadProps} onChange={onChange}>
            <Button icon={<UploadOutlined />} block>
              Click to Upload
            </Button>
          </Upload>
        </div>
      )}
    />
  );
};

const ConfigForm: React.FC = () => {
  const [trainedModels, setTrainedModels] = useState<Model[]>([]);
  const { videoMeta } = useVideo();
  const { requestJobs } = useJob();
  const formRef = useRef<ProFormInstance>();
  async function handleConfirm(formData: any) {
    if (formData.enableFlow && !formData.flowModel) {
      message.error("Please select a flow model!", 10);
      return false;
    }
    return userService
      .newTrackJob(
        videoMeta.id,
        formData.maxDet,
        formData.enableFlow,
        formData.animal,
        formData.segmModel,
        formData.poseModel,
        formData.flowModel
      )
      .then((res) => {
        message.success(res.data.message);
        requestJobs();
        return true;
      })
      .catch((e) => {
        console.log(e);
        const msg = e.response.data.message
          ? e.response.data.message
          : e.message;
        message.error(msg, 10);
      });
  }

  return videoMeta ? (
    <div
      style={{
        marginTop: 6,
      }}
    >
      <ModalForm
        // @ts-ignore
        labelWidth="auto"
        formRef={formRef}
        trigger={
          <Button block>
            <SendOutlined />
            Start Tracking
          </Button>
        }
        onFinish={handleConfirm}
        omitNil={false}
        title={videoMeta ? videoMeta.name : "Select a video"}
        onOpenChange={(open) => {
          if (open) {
            userService.getModels().then((res) => {
              let models = res.data as Model[];
              models = models.filter((m) => m.trained);
              setTrainedModels(models);
              console.log(res.data);
            });
          }
        }}
      >
        <ProFormDigit
          width="xs"
          name="maxDet"
          label="Maximum number of animals"
          min={1}
          rules={[
            {
              required: true,
              message: "Empty or invalid number of animals",
            },
          ]}
        />
        <ProFormSelect
          width="sm"
          name="animal"
          label="Animal"
          options={
            trainedModels.length > 0
              ? [
                  ...new Set(
                    trainedModels
                      .filter((m) => m.animal.length)
                      .map((m) => m.animal)
                  ),
                ]
              : []
          }
          rules={[
            {
              required: true,
              message: "Empty animal type!",
            },
          ]}
          onChange={(_) =>
            formRef.current.setFieldsValue({
              segmModel: undefined,
              poseModel: undefined,
            })
          }
        />
        <ProFormCheckbox
          tooltip="Optical flow generally makes tracking better, but requires more time to track"
          name="enableFlow"
          label="Enable Optical Flow"
          initialValue={false}
        />
        <ProFormDependency name={["animal", "enableFlow"]}>
          {({ animal, enableFlow }) => {
            return (
              <ProForm.Group>
                <ProFormSelect
                  width="md"
                  name="segmModel"
                  label="Instance Segmentation Model"
                  options={trainedModels
                    .filter((m) => m.type === "segm" && m.animal === animal)
                    .map((m) => ({
                      value: m.id,
                      label: m.id + ` (${m.method})`,
                    }))}
                  rules={[
                    {
                      required: true,
                      message: "Empty instance segmentation model!",
                    },
                  ]}
                />
                <ProFormSelect
                  width="md"
                  name="poseModel"
                  label="Pose Estimation Model"
                  options={trainedModels
                    .filter((m) => m.type === "pose" && m.animal === animal)
                    .map((m) => ({
                      value: m.id,
                      label: m.id + ` (${m.method})`,
                    }))}
                  rules={[
                    {
                      required: true,
                      message: "Empty pose estimation model!",
                    },
                  ]}
                />

                <ProFormSelect
                  width="md"
                  name="flowModel"
                  label="Optical Flow Model"
                  disabled={!enableFlow}
                  options={trainedModels
                    .filter((m) => m.type === "flow")
                    .map((m) => ({
                      value: m.id,
                      label: m.id + ` (${m.method})`,
                    }))}
                />
              </ProForm.Group>
            );
          }}
        </ProFormDependency>
      </ModalForm>
    </div>
  ) : (
    <></>
  );
};

const JobList: React.FC = () => {
  const [expandedRowKeys, setExpandedRowKeys] = useState<readonly Key[]>([]);
  const { requestJobs, jobs, setSelectedJob } = useJob();
  const rowSelection = {
    onChange: (selectedRowKeys: React.Key[], selectedRows: Job[]) => {
      console.log(
        `selectedRowKeys: ${selectedRowKeys}`,
        "selectedRows: ",
        selectedRows
      );
      setSelectedJob(selectedRows[0]);
    },
  };
  return (
    <ProList<Job>
      rowKey="taskId"
      headerTitle="Scheduled Jobs"
      pagination={{ pageSize: 5 }}
      dataSource={jobs}
      expandable={{ expandedRowKeys, onExpandedRowsChange: setExpandedRowKeys }}
      rowSelection={{ ...rowSelection, type: "radio" }}
      onItem={(record) => {
        return {
          onClick: () => {
            console.log(record);
          },
        };
      }}
      metas={{
        title: {
          dataIndex: "trueName",
        },
        content: {
          search: false,
          render: (_, _record) => (
            <div
              style={{
                minWidth: 200,
                flex: 1,
                display: "flex",
                justifyContent: "flex-end",
              }}
            ></div>
          ),
        },
        subTitle: {
          dataIndex: "status",
          render: (status: string, record) => (
            <Space>
              <Tag color={getStatusColor(status)}>{status.toUpperCase()}</Tag>
              <div
                style={{
                  width: "200px",
                }}
              >
                {record.status === "started" ? (
                  <Progress
                    percent={
                      record.progress
                        ? Math.round(
                            record.progress * 100 -
                              ((record.status === "started") as unknown as
                                | number
                                | 0)
                          )
                        : 0
                    }
                  />
                ) : null}
              </div>
            </Space>
          ),
        },
        description: {
          search: false,
          render: (_, record) => (
            <Descriptions
              column={{ xs: 1, sm: 1, md: 1, lg: 1, xl: 1, xxl: 1 }}
            >
              <Descriptions.Item label="Animal(s)">
                {record.maxDet}
              </Descriptions.Item>
              <Descriptions.Item label="Priority">
                {record.priority === 0 ? "Low" : "High"}
              </Descriptions.Item>
              <Descriptions.Item label="Created">
                {new Date(record.created).toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="Started">
                {record.started
                  ? new Date(record.started).toLocaleString()
                  : "Not yet"}
              </Descriptions.Item>
              <Descriptions.Item label="Finished">
                {record.ended
                  ? new Date(record.ended).toLocaleString()
                  : "Not yet"}
              </Descriptions.Item>
            </Descriptions>
          ),
        },
        avatar: {
          search: false,
          dataIndex: "trueName",
          render: (trueName: string) => (
            <Avatar
              style={{ backgroundColor: randomColor(trueName) }}
              size="small"
            >
              {trueName[0]}
            </Avatar>
          ),
        },
        actions: {
          search: false,
          render: (_, record) => {
            if (
              (record.status.toLowerCase() === "started" ||
                record.status.toLocaleLowerCase() === "queued") &&
              record.owned === true
            )
              return [
                <ButtonWithConfirmation
                  title="Cancel"
                  description="Are you sure you want to cancel this job?"
                  onConfirm={() => {
                    userService
                      .cancelJob(record.taskId)
                      .then((res) => {
                        message.success(res.data.message);
                        requestJobs();
                      })
                      .catch((e) => {
                        message.error(e.message, 10);
                      });
                  }}
                />,
              ];
            return [];
          },
        },
      }}
    />
  );
};
