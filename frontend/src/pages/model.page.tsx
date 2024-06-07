import React, { useState, useEffect } from "react";
import userService from "../services/user.service";
import { TableProps, Table, Tag, Tabs, message, Select, Tooltip } from "antd";
import ButtonWithConfirmation from "../components/ConfirmButton";
import { logError } from "../utils";
import TabContainer from "../components/TabContainer";
import { TabsProps } from "antd/lib";
import { Link } from "react-router-dom";

interface TrainDetJobType {
  taskId: string;
  trueName: string;
  datasetName: string;
  status: string;
  created: string;
  started: string;
  ended: string;
  priority: number;
  owned: boolean;
  config: string;
  valRatio: number;
  modelName: string;
}

const DetTrainTable: React.FC = () => {
  const [jobs, setJobs] = useState<TrainDetJobType[]>([]);
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout>(null); // For polling jobs
  const requestJobs = () => {
    userService
      .getDetTrainJobs()
      .then((res) => {
        setJobs(res.data as TrainDetJobType[]);
        console.log("det jobs", res.data);
        clearTimeout(timeoutId);
        setTimeoutId(setTimeout(requestJobs, 1000 * 10));
      })
      .catch(logError);
  };

  const detColumns: TableProps<TrainDetJobType>["columns"] = [
    {
      title: "User Name",
      dataIndex: "trueName",
      key: "name",
    },
    { title: "Model Name", dataIndex: "modelName", key: "modelName" },
    {
      title: "Dataset Name",
      dataIndex: "datasetName",
      key: "datasetName",
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
    },
    {
      title: "Created",
      dataIndex: "created",
      key: "created",
      sorter: (a, b) => (a.created > b.created ? 1 : -1),
      sortDirections: ["descend", "ascend"],
      render: (value) => (value ? new Date(value).toLocaleString() : null),
    },
    {
      title: "Started",
      dataIndex: "started",
      key: "started",
      sorter: (a, b) => (a.started > b.started ? 1 : -1),
      sortDirections: ["descend", "ascend"],
      render: (value) => (value ? new Date(value).toLocaleString() : null),
    },
    {
      title: "Ended",
      dataIndex: "ended",
      key: "ended",
      sorter: (a, b) => (a.ended > b.ended ? 1 : -1),
      sortDirections: ["descend", "ascend"],
      render: (value) => (value ? new Date(value).toLocaleString() : null),
    },
    {
      title: "Priority",
      dataIndex: "priority",
      key: "priority",
    },
    {
      title: "Config",
      dataIndex: "config",
      key: "config",
    },
    {
      title: "Val Ratio",
      dataIndex: "valRatio",
      key: "valRatio",
    },
    {
      title: "Action",
      key: "action",
      render: (_, record) =>
        record.owned &&
        (record.status === "started" || record.status === "queued") ? (
          <ButtonWithConfirmation
            onConfirm={() => {
              userService.cancelJob(record.taskId).then((response) => {
                message.success(response.data.message);
                requestJobs();
              });
            }}
            title="Cancel"
            description={
              <>
                <p>Are you sure you want to cancel this job?</p>
              </>
            }
          />
        ) : null,
    },
  ];
  useEffect(() => {
    requestJobs();
    return () => {
      clearTimeout(timeoutId);
    };
  }, []);
  return <Table columns={detColumns} dataSource={jobs}></Table>;
};

interface TrainPoseJobType extends TrainDetJobType {
  links: string[][];
  swaps: string[][];
}

const PoseTrainTable: React.FC = () => {
  const [jobs, setJobs] = useState<TrainPoseJobType[]>([]);
  const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout>(null); // For polling jobs
  const requestJobs = () => {
    userService
      .getPoseTrainJobs()
      .then((res) => {
        setJobs(res.data as TrainPoseJobType[]);
        console.log("pose jobs", res.data);
        clearTimeout(timeoutId);
        setTimeoutId(setTimeout(requestJobs, 1000 * 10));
      })
      .catch(logError);
  };

  const poseColumns: TableProps<TrainPoseJobType>["columns"] = [
    {
      title: "User Name",
      dataIndex: "trueName",
      key: "name",
    },
    { title: "Model Name", dataIndex: "modelName", key: "modelName" },
    {
      title: "Dataset Name",
      dataIndex: "datasetName",
      key: "datasetName",
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
    },
    {
      title: "Created",
      dataIndex: "created",
      key: "created",
      sorter: (a, b) => (a.created > b.created ? 1 : -1),
      sortDirections: ["descend", "ascend"],
      render: (value) => (value ? new Date(value).toLocaleString() : null),
    },
    {
      title: "Started",
      dataIndex: "started",
      key: "started",
      sorter: (a, b) => (a.started > b.started ? 1 : -1),
      sortDirections: ["descend", "ascend"],
      render: (value) => (value ? new Date(value).toLocaleString() : null),
    },
    {
      title: "Ended",
      dataIndex: "ended",
      key: "ended",
      sorter: (a, b) => (a.ended > b.ended ? 1 : -1),
      sortDirections: ["descend", "ascend"],
      render: (value) => (value ? new Date(value).toLocaleString() : null),
    },
    {
      title: "Priority",
      dataIndex: "priority",
      key: "priority",
    },
    {
      title: "Config",
      dataIndex: "config",
      key: "config",
    },
    {
      title: "Val Ratio",
      dataIndex: "valRatio",
      key: "valRatio",
    },
    {
      title: "Swaps",
      dataIndex: "swaps",
      key: "swaps",
      render: (swaps) => (
        <>
          {swaps.map((link, i) => (
            <Tag key={i}>{link.join("<>")}</Tag>
          ))}
        </>
      ),
    },
    {
      title: "Links",
      dataIndex: "links",
      key: "links",
      render: (links) => (
        <>
          {links.map((link, i) => (
            <Tag key={i}>{link.join("->")}</Tag>
          ))}
        </>
      ),
    },
    {
      title: "Action",
      key: "action",
      render: (_, record) =>
        record.owned &&
        (record.status === "started" || record.status === "queued") ? (
          <ButtonWithConfirmation
            title="Cancel"
            onConfirm={() => {
              userService
                .cancelJob(record.taskId)
                .then((response) => {
                  message.success(response.data.message);
                  requestJobs();
                })
                .catch(logError);
            }}
            description={
              <>
                <p>Are you sure you want to cancel this job?</p>
              </>
            }
          />
        ) : null,
    },
  ];
  useEffect(() => {
    requestJobs();
    return () => {
      clearTimeout(timeoutId);
    };
  }, []);
  return <Table columns={poseColumns} dataSource={jobs}></Table>;
};

interface Model {
  id: string;
  name: string;
  config: string;
  checkpoint: string;
  animal: string;
  type: string;
  method: string;
  owned: boolean;
  trained: boolean;
}

const ModelPage = () => {
  const items: TabsProps["items"] = [
    { key: "detection", label: "Detection", children: <DetTrainTable /> },
    { key: "pose", label: "Pose", children: <PoseTrainTable /> },
  ];
  const [models, setModels] = useState<Model[]>([]);
  const requestModels = () => {
    userService
      .getModels()
      .then((res) => {
        console.log(res.data);
        setModels(res.data as Model[]);
      })
      .catch(logError);
  };
  useEffect(() => {
    requestModels();
  }, []);

  const handleModelStatusChange = (value: Boolean, record: Model) => {
    userService
      .updateModel(record.id, value)
      .then((response) => {
        message.success(response.data.message);
        requestModels();
      })
      .catch(logError);
  };

  const modelColumns: TableProps<Model>["columns"] = [
    {
      title: "Model Name",
      dataIndex: "name",
      key: "name",
      render(value, record) {
        return record.owned ? (
          <Link to={`/models/${record.id}`}>
            <a>{value}</a>
          </Link>
        ) : (
          value
        );
      },
    },
    {
      title: "Config",
      dataIndex: "config",
      key: "config",
      ellipsis: true,
    },
    {
      title: "Checkpoint",
      dataIndex: "checkpoint",
      key: "checkpoint",
      ellipsis: true,
    },
    {
      title: "Animal",
      dataIndex: "animal",
      key: "animal",
    },
    {
      title: "Type",
      dataIndex: "type",
      key: "type",
    },
    {
      title: "Method",
      dataIndex: "method",
      key: "method",
      ellipsis: true,
    },
    {
      title: "Trained",
      dataIndex: "trained",
      key: "trained",
      render: (_, record) => {
        if (!record.owned) return record.trained ? "Yes" : "No";
        const options = [
          { label: "Yes", value: true },
          { label: "No", value: false },
        ];
        return (
          <Tooltip title="You may modify the status of the model so that you can use it in tracking">
            <Select
              options={options}
              value={record.trained}
              onChange={(value) => handleModelStatusChange(value, record)}
            />
          </Tooltip>
        );
      },
    },
    {
      title: "Action",
      key: "action",
      render: (_, record) =>
        record.owned ? (
          <ButtonWithConfirmation
            onConfirm={() => {
              userService
                .deleteModel(record.id)
                .then((response) => {
                  message.success(response.data.message);
                  requestModels();
                })
                .catch(logError);
            }}
            description={
              <>
                <p>Are you sure you want to delete this model?</p>
              </>
            }
          />
        ) : null,
    },
  ];
  return (
    <>
      <TabContainer title="Available Models">
        <Table columns={modelColumns} dataSource={models}></Table>
      </TabContainer>
      <TabContainer title="Training Models">
        <Tabs items={items}></Tabs>
      </TabContainer>
    </>
  );
};

export default ModelPage;
