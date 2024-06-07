import React, { useEffect, useState } from "react";
import userService from "../services/user.service";
import {
  Button,
  Card,
  List,
  message,
  Select,
  Form,
  Modal,
  Input,
  Descriptions,
  InputNumber,
  TreeSelect,
  Space,
} from "antd";
import type { ButtonProps } from "antd";
import { logError } from "../utils";
import { Link, useNavigate } from "react-router-dom";
import { useDataset, type Dataset } from "../context/datasetProvider";
import ButtonWithConfirmation from "../components/ConfirmButton";

type NewDatasetField = {
  name: string;
  keypoints: string[];
  animalName: string;
};

const NewDatasetButton: React.FC<ButtonProps> = (props: ButtonProps) => {
  return (
    <div
      style={{
        textAlign: "center",
        marginTop: 12,
        height: 32,
        lineHeight: "32px",
      }}
    >
      <Button {...props}>Create New Dataset</Button>
    </div>
  );
};

const DatasetPage: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [isNewDatasetConfigOpen, setNewDatasetConfigOpen] = useState(false);
  const [newDatasetForm] = Form.useForm();
  const { setDataset } = useDataset();
  const [modelTrainerOpen, setModelTrainerOpen] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null);

  const handleOk = () => {
    newDatasetForm.validateFields().then(({ name, animalName, keypoints }) => {
      console.log(name, animalName, keypoints);
      userService
        .createNewDataset(name, keypoints, animalName)
        .then((response) => {
          message.success(response.data);
          setNewDatasetConfigOpen(false);
          updateDatasets();
        })
        .catch((e) => {
          logError(e);
          setNewDatasetConfigOpen(false);
        });
    });
  };

  const updateDatasets = () => {
    userService
      .getDatasets()
      .then((response) => {
        setDatasets(response.data);
      })
      .catch(logError);
  };
  const handleDeleteDataset = (datasetId: number) => {
    userService
      .deleteDataset(datasetId)
      .then((response) => {
        message.success(response.data.message);
        updateDatasets();
      })
      .catch(logError);
  };

  useEffect(() => {
    updateDatasets();
  }, []);

  return (
    <>
      <Modal
        title="Basic Modal"
        open={isNewDatasetConfigOpen}
        onOk={handleOk}
        onCancel={() => {
          setNewDatasetConfigOpen(false);
        }}
      >
        <Form form={newDatasetForm}>
          <Form.Item<NewDatasetField>
            label="Animal Name"
            name="animalName"
            rules={[{ required: true }]}
          >
            <Input />
          </Form.Item>
          <Form.Item<NewDatasetField>
            label="Keypoints"
            name="keypoints"
            rules={[{ required: true }]}
          >
            <Select
              mode="tags"
              placeholder="Type in a keypoint name and press enter"
            />
          </Form.Item>
          <Form.Item<NewDatasetField>
            label="Name"
            name="name"
            rules={[{ required: true }]}
          >
            <Input />
          </Form.Item>
        </Form>
      </Modal>
      {selectedDataset ? (
        <ModelTrainer
          open={modelTrainerOpen}
          setOpen={setModelTrainerOpen}
          dataset={selectedDataset}
        />
      ) : null}
      <Space direction="vertical">
        <List
          grid={{
            gutter: 16,
            xs: 1,
            sm: 2,
            md: 4,
            lg: 4,
            xl: 6,
            xxl: 3,
          }}
          loadMore={
            <NewDatasetButton onClick={() => setNewDatasetConfigOpen(true)} />
          }
          pagination={{ position: "bottom", align: "end" }}
          dataSource={datasets}
          renderItem={(dataset) => (
            <List.Item>
              <Card
                title={dataset.name}
                actions={[
                  <Link
                    to={`/datasets/${dataset.id}`}
                    onClick={() => setDataset(dataset)}
                  >
                    <Button
                      type="link"
                      onClick={() => {
                        setDataset(dataset);
                      }}
                    >
                      View
                    </Button>
                  </Link>,
                  <Button
                    type="link"
                    onClick={() => {
                      setSelectedDataset(dataset);
                      setModelTrainerOpen(true);
                    }}
                  >
                    Train
                  </Button>,
                  <ButtonWithConfirmation
                    onConfirm={() => {
                      handleDeleteDataset(dataset.id);
                    }}
                    description={
                      <>
                        <p>Are you sure you wanna delete this dataset?</p>
                        <p>This operation cannot be reversed!</p>
                      </>
                    }
                  />,
                ]}
              >
                <Descriptions column={2}>
                  <Descriptions.Item label="Animal Name" span={1}>
                    {dataset.animalName}
                  </Descriptions.Item>
                  <Descriptions.Item label="Keypoints" span={1}>
                    {dataset.keypoints.join(", ")}
                  </Descriptions.Item>
                  <Descriptions.Item label="Images" span={1}>
                    {dataset.images}
                  </Descriptions.Item>
                  <Descriptions.Item label="Annotations" span={1}>
                    {dataset.annotations}
                  </Descriptions.Item>
                  <Descriptions.Item label="Created at" span={2}>
                    {dataset.created}
                  </Descriptions.Item>
                </Descriptions>
              </Card>
            </List.Item>
          )}
        />
      </Space>
    </>
  );
};

type ModelTrainerProps = {
  open: boolean;
  setOpen: (open: boolean) => void;
  dataset: Dataset;
};

const ModelTrainer: React.FC<ModelTrainerProps> = ({
  open,
  setOpen,
  dataset,
}) => {
  const [configs, setConfigs] = useState([]);
  const [selectedConfig, setSelectedConfig] = useState<string | null>(null);
  const [choices, setChoices] = useState([]);
  const [symKptsChoices, setSymKptsChoices] = useState([]);
  const [skeletonChoices, setSkeletonChoices] = useState([]);
  const [form] = Form.useForm();
  const navigate = useNavigate();
  useEffect(() => {
    userService
      .getModelConfigs()
      .then((response) => {
        setConfigs(response.data);
      })
      .catch(logError);
  }, []);
  useEffect(() => {
    if (selectedConfig && selectedConfig.includes("pose")) {
      let kptsPair = [];
      const kpts = dataset.keypoints;
      for (let i = 0; i < kpts.length; i++) {
        kptsPair.push({
          title: kpts[i],
          value: kpts[i],
          selectable: false,
          children: [],
        });
        for (let j = 0; j < kpts.length; j++) {
          const value = `${kpts[i]}-${kpts[j]}`;
          if (i === j) continue;
          kptsPair[kptsPair.length - 1].children.push({
            title: value,
            value: `${kpts[i]}-${kpts[j]}`,
          });
        }
      }
      setChoices(kptsPair);
    }
  }, [dataset, selectedConfig]);

  const handleOk = (dataset) => {
    form
      .validateFields()
      .then((params) => {
        console.log(params);
        const { config } = params;
        console.log(params);
        if (config && config.includes("pose")) {
          let parsedLinks = [];
          let parsedSwaps = [];
          const kpts = dataset.keypoints;
          for (let i = 0; i < params.links.length; i++) {
            let link = [];
            for (let j = 0; j < kpts.length; j++) {
              if (params.links[i].includes(kpts[j])) link.push(kpts[j]);
            }
            parsedLinks.push(link);
          }
          for (let i = 0; i < params.swaps.length; i++) {
            let swap = [];
            for (let j = 0; j < kpts.length; j++) {
              if (params.swaps[i].includes(kpts[j])) swap.push(kpts[j]);
            }
            parsedSwaps.push(swap);
          }
          userService
            .newPoseTrainJob(
              config,
              dataset.id,
              params.valRatio,
              params.modelName,
              parsedLinks,
              parsedSwaps
            )
            .then((response) => {
              message.success(response.data.message);
              setOpen(false);
              navigate("/models");
            })
            .catch((e) => {
              logError(e);
            });
        } else {
          userService
            .newDetTrainJob(
              params.config,
              dataset.id,
              params.valRatio,
              params.modelName
            )
            .then((response) => {
              message.success(response.data.message);
              setOpen(false);
              navigate("/models");
            })
            .catch(logError);
        }
      })
      .catch((e) => console.log(e));
  };

  return (
    <Modal
      title={`Train model for ${dataset.name}`}
      open={open}
      onCancel={() => setOpen(false)}
      onOk={() => handleOk(dataset)}
    >
      <Form form={form} labelCol={{ span: 9 }} wrapperCol={{ span: 16 }}>
        <Form.Item
          label="Model name"
          name="modelName"
          rules={[{ required: true }]}
        >
          <Input />
        </Form.Item>
        <Form.Item
          label="Model Config"
          name="config"
          rules={[{ required: true }]}
        >
          <TreeSelect onChange={setSelectedConfig} treeData={configs} />
        </Form.Item>
        <Form.Item
          label="Val ratio"
          name="valRatio"
          rules={[{ required: true }]}
        >
          <InputNumber min={0.1} max={0.5} step={0.1} />
        </Form.Item>
        {selectedConfig && selectedConfig.includes("pose") ? (
          <>
            <Form.Item
              label="Symmetry keypoints"
              name="swaps"
              rules={[{ required: true }]}
            >
              <TreeSelect
                value={symKptsChoices}
                onChange={setSymKptsChoices}
                treeData={choices}
                multiple
              />
            </Form.Item>
            <Form.Item
              label="Skeleton"
              name="links"
              rules={[{ required: true }]}
            >
              <TreeSelect
                value={skeletonChoices}
                onChange={setSkeletonChoices}
                treeData={choices}
                multiple
              />
            </Form.Item>
          </>
        ) : null}
      </Form>
    </Modal>
  );
};

export default DatasetPage;
