import React, { useEffect, useState, useRef } from "react";
import Resizer from "../Components/Resizer/Resizer";
import {
  Pagination,
  Empty,
  InputNumber,
  Button,
  Space,
  Popover,
  Select,
  Input,
  message,
  Spin,
  Tabs,
  Table,
  Cascader,
  Tag,
  Divider,
} from "antd";
import {
  DeleteOutlined,
  SettingOutlined,
  SaveOutlined,
  ForwardOutlined,
  PlusOutlined,
} from "@ant-design/icons";
import TabContainer from "../Components/TabContainer/TabContainer";
import { connect } from "react-redux";
import global from "../config";
import axios from "axios";
import { randomColor } from "../utils";
import { HexColorPicker } from "react-colorful";

const getSVGcoords = (e, svg) => {
  const pt = svg.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
};

export const AnnotatorAndTrainer = (props) => {
  return (
    <Resizer style={{ width: "100%" }}>
      <TabContainer title="Annotator">
        <Tabs defaultActiveKey="1">
          <Tabs.TabPane tab="Annotator" key="1">
            <VisibleAnnotator />
          </Tabs.TabPane>
          <Tabs.TabPane tab="Datasets" key="2">
            <Trainer />
          </Tabs.TabPane>
        </Tabs>
      </TabContainer>
    </Resizer>
  );
};

const Trainer = (props) => {
  const [selectedRow, setSelectedRow] = useState({});
  const [modelName, setModelName] = useState("");
  const [connections, setConnections] = useState([]);
  const [mirrorKeypoints, setMirrorKeypoints] = useState({
    left: [],
    right: [],
  });
  const [cascadeOptions, setCascadeOptions] = useState([]);
  const [animalNames, setAnimalNames] = useState([]);
  const [selectedAnimal, setSelectedAnimal] = useState("");
  const [datasets, setDatasets] = useState({});

  useEffect(() => {
    axios.get("animals").then((res) => {
      const animals = res.data.animals;
      console.log(animals);
      setAnimalNames(animals);
      axios.get("datasets").then((res) => {
        const datasets = res.data;
        for (let [_, ds] of Object.entries(datasets))
          ds.forEach((x) => (x.key = x.datasetName));
        setDatasets(res.data);
      });
    });
  }, []);

  const downloadDataset = (datasetName) => {
    axios
      .get(`datasets/download/${datasetName}`)
      .then((res) => {
        if (res.data.status === "success") {
          let a = document.createElement("a");
          a.href = res.data.src;
          a.setAttribute("download", `${datasetName}.json`);
          document.body.appendChild(a);
          a.click();
          a.remove();
        } else message.error(res.data.message);
      })
      .catch((e) => {
        console.log(e);
        message.error("internal error");
      });
  };

  const TABLE_COLUMNS = [
    { title: "Dataset", dataIndex: "datasetName" },
    {
      title: "bodyparts",
      dataIndex: "keypoints",
      render: (bodyparts) => {
        return bodyparts.map((bodypart) => <Tag>{bodypart}</Tag>);
      },
    },
    { title: "Instances", dataIndex: "numInstances" },
    {
      title: "Actions",
      key: "action",
      render: (text) => {
        return (
          <Space>
            <a onClick={() => downloadDataset(text.datasetName)}>Download</a>
            <a>Delete</a>
          </Space>
        );
      },
    },
  ];

  const rowSelection = {
    onChange: (_, selectedRows) => {
      console.log(selectedRows);
      if (selectedRows.length > 0) {
        setSelectedRow(selectedRows[0]);
      } else setSelectedRow({});
    },
    type: "radio",
  };

  const handleConnectionsSelection = (value, bodyparts) => {
    if (!bodyparts) return;
    const connsMat = new Array(bodyparts.length);
    const conns = [];
    for (let i = 0; i < bodyparts.length; i++)
      connsMat[i] = new Array(bodyparts.length).fill(false);
    for (let item of value) {
      const connection = item.slice(-1)[0].split("-");
      const source = bodyparts.indexOf(connection[0]);
      if (connection.length === 1) {
        for (let i = 0; i < bodyparts.length; i++) {
          if (i === source) continue;
          connsMat[source][i] = true;
        }
      } else if (connection.length === 2) {
        const target = bodyparts.indexOf(connection[1]);
        connsMat[source][target] = true;
      }
    }
    for (let i = 0; i < bodyparts.length; i++) {
      for (let j = 0; j < bodyparts.length; j++) {
        if (i === j) continue;
        if (connsMat[i][j]) conns.push([i, j]);
      }
    }
    setConnections(conns);
  };

  const handleMirrorKeypointsSelection = (value, place, mirrorKeypoints) => {
    value = value.map((x) => Math.floor(x));
    switch (place) {
      case "left":
        setMirrorKeypoints({ left: value, right: mirrorKeypoints.right });
        return;
      case "right":
        setMirrorKeypoints({ left: mirrorKeypoints.left, right: value });
        return;
      default:
        return;
    }
  };

  useEffect(() => {
    if (!Object.keys(selectedRow).length) return;
    const bodyparts = selectedRow.keypoints;
    const newOptions = [];
    for (let bodypart of bodyparts) {
      const children = [];
      for (let anotherBodypart of bodyparts) {
        if (anotherBodypart === bodypart) continue;
        children.push({
          value: `${bodypart}-${anotherBodypart}`,
          label: `${bodypart}-${anotherBodypart}`,
        });
      }
      newOptions.push({ value: bodypart, label: bodypart, children });
    }
    console.log(newOptions);
    setCascadeOptions(newOptions);
  }, [selectedRow]);

  const startTraining = (
    datasetName,
    connections,
    modelName,
    mirrorKeypoints
  ) => {
    if (mirrorKeypoints.left.length !== mirrorKeypoints.right.length) {
      message.error("Please select the same number of keypoints on both sides");
      return;
    }
    for (let lKpt of mirrorKeypoints.left) {
      if (mirrorKeypoints.right.indexOf(lKpt) !== -1) {
        message.error("Cannot have the same keypoint on both sides");
        return;
      }
    }
    let body = {
      connections,
      datasetName,
      modelName,
      mirrorKeypoints,
    };
    console.log(body);

    axios.post("/train", body).then((res) => console.log(res.data));
  };
  return (
    <>
      <Select
        placeholder="Pick an animal"
        style={{ width: 240 }}
        onChange={setSelectedAnimal}
      >
        {animalNames.map((x) => (
          <Select.Option value={x}>{x}</Select.Option>
        ))}
      </Select>
      <Table
        rowSelection={rowSelection}
        columns={TABLE_COLUMNS}
        dataSource={datasets[selectedAnimal]}
      />
      <div style={{ textAlign: "center" }}>
        <Space direction="vertical">
          <Space>
            <span>Keypoint connections: </span>
            <Cascader
              options={cascadeOptions}
              multiple
              changeOnSelect
              onChange={(value, _) =>
                handleConnectionsSelection(value, selectedRow.keypoints)
              }
              style={{ minWidth: 400 }}
              placeholder="Choose keypoints connections"
            />
          </Space>
          <Space>
            <span>Mirror keypoints: </span>
            <Select
              placeholder="left"
              mode="multiple"
              allowClear
              defaultValue={mirrorKeypoints.left}
              onChange={(value) =>
                handleMirrorKeypointsSelection(value, "left", mirrorKeypoints)
              }
              style={{ minWidth: 160 }}
            >
              {selectedRow.keypoints
                ? selectedRow.keypoints.map((x, i) => (
                    <Select.Option key={i}>{x}</Select.Option>
                  ))
                : null}
            </Select>
            <Select
              placeholder="right"
              mode="multiple"
              allowClear
              defaultValue={mirrorKeypoints.right}
              style={{ minWidth: 160 }}
              onChange={(value) =>
                handleMirrorKeypointsSelection(value, "right", mirrorKeypoints)
              }
            >
              {selectedRow.keypoints
                ? selectedRow.keypoints.map((x, i) => (
                    <Select.Option key={i}>{x}</Select.Option>
                  ))
                : null}
            </Select>
          </Space>
          <Space>
            <Input
              placeholder="input model name"
              onChange={(e) => setModelName(e.target.value)}
              style={{ minWidth: 200 }}
              addonBefore="Model name"
            />
            <Button
              type="primary"
              onClick={() =>
                startTraining(
                  selectedRow.datasetName,
                  connections,
                  modelName,
                  mirrorKeypoints
                )
              }
              icon={<ForwardOutlined />}
            >
              Train
            </Button>
          </Space>
        </Space>
      </div>
    </>
  );
};

const EmptyAnnotator = (props) => {
  const [fetchFrames, setFetchFrames] = useState(50);
  const [options, setOptions] = useState([]);
  const [videoName, setVideoName] = useState("");

  useEffect(() => {
    axios.get("videos").then((res) => {
      const videos = res.data.videos;
      const ops = videos.map((x) => (
        <Select.Option value={x.fname}>{x.fname}</Select.Option>
      ));
      setOptions(ops);
    });
  }, []);

  const component = (
    <Space>
      <Select
        style={{ width: 240 }}
        onChange={setVideoName}
        placeholder="Pick a video"
      >
        {options}
      </Select>
      <InputNumber
        min={1}
        defaultValue={50}
        onChange={setFetchFrames}
        value={fetchFrames}
      />
      <Button
        type="primary"
        onClick={() =>
          props.createMoreData(videoName, fetchFrames, props.selectedAnimal)
        }
      >
        Create Now
      </Button>
    </Space>
  );

  return props.plain ? (
    component
  ) : (
    <Empty description={<span>Add some more data?</span>}>
      {props.selectedAnimal ? component : null}
    </Empty>
  );
};

const AnnotatorWrapper = (props) => {
  const [imgList, setImgList] = useState([]);
  const [animalNames, setAnimalNames] = useState([]);
  const [selectedAnimal, setSelectedAnimal] = useState(null);
  const animalRef = useRef();
  animalRef.current = selectedAnimal;
  const [datasets, setDatasets] = useState({});
  const datasetsRef = useRef();
  datasetsRef.current = datasets;
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [newDatasetName, setNewDatasetName] = useState("");
  const newDatasetNameRef = useRef();
  newDatasetNameRef.current = newDatasetName;
  const [bodyparts, setBodyparts] = useState([]);
  const bodypartsRef = useRef();
  bodypartsRef.current = bodyparts;
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    axios.get("animals").then((res) => {
      const animals = res.data.animals;
      setAnimalNames(animals);
    });
  }, []);

  useEffect(() => {
    axios.get("datasets").then((res) => {
      const data = res.data;
      const newDatasets = {};
      for (let animal of Object.keys(data)) {
        newDatasets[animal] = {};
        for (let dataset of data[animal]) {
          newDatasets[animal][dataset.datasetName] = dataset;
        }
      }
      setDatasets(newDatasets);
    });
  }, [animalNames]);

  const createNewDataset = () => {
    const datasetName = newDatasetNameRef.current;
    const bodyparts = bodypartsRef.current;
    const animalName = animalRef.current;
    console.log(datasetName, bodyparts, animalName);
    if (datasetName === "") {
      message.error("Need dataset name");
      return;
    }
    if (bodyparts.length === 0) {
      message.error("Need bodyparts");
      return;
    }
    axios
      .post("datasets/new", {
        datasetName,
        keypoints: bodyparts,
        animalName,
      })
      .then((res) => {
        if (res.data.status === "success") {
          message.success(res.data.message);
          axios.get("datasets").then((res) => {
            const data = res.data;
            const newDatasets = {};
            for (let animal in Object.keys(data)) {
              newDatasets[animal] = {};
              for (let dataset in data[animal])
                newDatasets[animal][dataset.datasetName] = dataset;
            }
            setDatasets(newDatasets);
          });
        } else message.error(res.data.message);
        setNewDatasetName("");
        setBodyparts([]);
      })
      .catch((e) => {
        console.log(e);
        message.error("internal error");
      });
  };

  function createMoreData(videoName, frames, selectedAnimal) {
    if (!selectedAnimal) {
      message.error("Please select an animal!");
      return;
    }
    let body = {
      frames: frames,
      videoName: videoName,
      animalName: selectedAnimal,
      cropArea: props.cropArea,
      cropTime: props.cropTime,
    };
    console.log(body);
    setLoading(true);
    axios
      .post("/annotator/more", body)
      .then((res) => {
        let data = res.data;
        if (data.status === "success") {
          setImgList(data.result);
          message.success("successfully created images");
        } else message.error(data.message);
      })
      .catch((e) => {
        console.log(e);
      });
    setLoading(false);
  }

  useEffect(() => {
    setSelectedDataset(null);
    if (selectedAnimal) {
      axios.post("/images", { animalName: selectedAnimal }).then((res) => {
        if (res.data.result.length === 0)
          message.warning("No image found. You may create some new images.");
        console.log(res);
        setImgList(res.data.result);
      });
    }
  }, [selectedAnimal]);

  const dropdownRender = (menus) => {
    return (
      <div style={{ textAlign: "center" }}>
        {menus}
        <Divider />
        <Space direction="vertical">
          <Input
            placeholder="New dataset name"
            onChange={(e) => setNewDatasetName(e.target.value)}
            value={newDatasetName}
          />
          <Select
            mode="tags"
            placeholder="Add Bodypart (space to separate)"
            style={{ minWidth: 240, textAlign: "left" }}
            onChange={setBodyparts}
            defaultValue={bodyparts}
            value={bodyparts}
            tokenSeparators={[" "]}
          >
            {bodyparts.map((x) => (
              <Select.Option value={x}>{x}</Select.Option>
            ))}
          </Select>
          <Button
            type="text"
            icon={<PlusOutlined />}
            onClick={createNewDataset}
          >
            Create
          </Button>
        </Space>
      </div>
    );
  };

  const selectComponent = (
    <Space>
      <Select
        onChange={setSelectedAnimal}
        placeholder="Pick an animal"
        style={{ minWidth: 120 }}
        value={selectedAnimal}
      >
        {animalNames.map((x) => (
          <Select.Option value={x}>{x}</Select.Option>
        ))}
      </Select>
      <Select
        onChange={setSelectedDataset}
        placeholder="Choose a dataset"
        style={{ minWidth: 260 }}
        disabled={!selectedAnimal}
        dropdownRender={dropdownRender}
        value={selectedDataset}
      >
        {datasets[selectedAnimal]
          ? Object.keys(datasets[selectedAnimal]).map((x) => (
              <Select.Option value={x}>{x}</Select.Option>
            ))
          : []}
      </Select>
    </Space>
  );

  return (
    <div style={{ textAlign: "center" }}>
      <Space direction="vertical">
        <Spin spinning={loading}>
          {imgList.length <= 0 ? (
            <EmptyAnnotator
              selectedAnimal={selectedAnimal}
              createMoreData={createMoreData}
              plain={false}
            />
          ) : (
            <>
              <Annotator
                imgList={imgList}
                bodyparts={
                  datasets[selectedAnimal] &&
                  datasets[selectedAnimal][selectedDataset]
                    ? datasets[selectedAnimal][selectedDataset].keypoints
                    : []
                }
                datasetName={selectedDataset}
                selectedAnimal={selectedAnimal}
                createMoreData={createMoreData}
              >
                {selectComponent}
              </Annotator>
            </>
          )}
        </Spin>
        {imgList.length <= 0 ? selectComponent : null}
      </Space>
    </div>
  );
};

const Annotator = (props) => {
  const [currImgIdx, setCurrImgIdx] = useState(0);
  const [objNum, setObjNum] = useState(2);
  const [colors, setColors] = useState([]);
  const [circleRadius, setCircleRadius] = useState(3);
  const [kptVisibility, setKptVisibility] = useState({});

  const svg = useRef(null);
  const imgRef = useRef(null);
  const [markers, setMarkers] = useState([]);
  const markersTable = useRef({});

  const colorsRef = useRef([]);
  const bodypartsRef = useRef([]);
  const rRef = useRef();
  const markersRef = useRef([]);
  const objNumRef = useRef();
  const currImgIdxRef = useRef();

  markersRef.current = markers;
  colorsRef.current = colors;
  bodypartsRef.current = props.bodyparts;
  rRef.current = circleRadius;
  objNumRef.current = objNum;
  currImgIdxRef.current = currImgIdx;

  const saveAnnotations = (
    imgList,
    lenBodyparts,
    objNum,
    markers,
    currImgIdx,
    datasetName
  ) => {
    markersTable.current[currImgIdx] = [...markers];
    const result = {};
    for (const [key, list] of Object.entries(markersTable.current)) {
      const img = imgList[key];
      result[img.image_id] = [];
      const allKeypoints = [];
      for (const marker of list) {
        allKeypoints.push(marker.cx);
        allKeypoints.push(marker.cy);
        if (marker.visible) allKeypoints.push(2);
        else allKeypoints.push(1);
      }
      const kptArr = [];
      while (allKeypoints.length)
        kptArr.push(allKeypoints.splice(0, lenBodyparts * 3));
      for (let i = 0; i < kptArr.length; i++) {
        if (kptArr[i].length < lenBodyparts * 3) break;
        result[img.image_id].push(kptArr[i]);
      }
    }
    console.log(result);
    axios
      .post("datasets/append", { datasetName, data: result })
      .then((res) => {
        if (res.data.status === "success") message.success(res.data.message);
        else message.error(res.data.message);
      })
      .catch((e) => {
        console.log(e);
        message.error("internal error");
      });
  };

  useEffect(() => {
    const handleMouseDown = (e) => {
      e.preventDefault();
      switch (e.which) {
        case 1:
          if (
            markersRef.current.length >=
            bodypartsRef.current.length * objNumRef.current
          )
            break;
          const pt = getSVGcoords(e, svg.current);
          setMarkers([
            ...markersRef.current,
            {
              cx: pt.x,
              cy: pt.y,
              r: rRef.current,
              color:
                colorsRef.current[
                  Math.floor(
                    markersRef.current.length / bodypartsRef.current.length
                  )
                ],
              id: markersRef.current.length,
              visible: true,
              setVisibility: setKptVisibility,
              bodypart:
                bodypartsRef.current[
                  markersRef.current.length % bodypartsRef.current.length
                ],
            },
          ]);
          break;
        case 3:
          setMarkers(markersRef.current.slice(0, -1));
          break;
        default:
          break;
      }
    };

    const svgHandle = svg.current;
    svgHandle.addEventListener("mousedown", handleMouseDown);
    svgHandle.addEventListener("contextmenu", (e) => {
      e.preventDefault();
    });
    return () => {
      svgHandle.removeEventListener("mousedown", handleMouseDown);
      svgHandle.removeEventListener("contextmenu", (e) => {
        e.preventDefault();
      });
    };
  }, []);

  useEffect(() => {
    if (!props.datasetName) return;
    setMarkers([]);
    markersTable.current = {};
    axios.get(`annotations/${props.datasetName}`).then((res) => {
      const annotations = res.data;
      const imgNameIdxDict = {};
      props.imgList.forEach((x, i) => (imgNameIdxDict[x.name] = i));
      for (let imgName of Object.keys(annotations))
        markersTable.current[imgNameIdxDict[imgName]] = [];
      for (let [imgName, kptArr] of Object.entries(annotations)) {
        const idx = imgNameIdxDict[imgName];
        for (let keypoints of kptArr) {
          for (let i = 0; i < keypoints.length; i += 3) {
            const marker = {
              cx: keypoints[i],
              cy: keypoints[i + 1],
              r: rRef.current,
              color:
                colorsRef.current[
                  Math.floor(
                    markersTable.current[idx].length /
                      bodypartsRef.current.length
                  )
                ],
              id: markersTable.current[idx].length,
              visible: keypoints[i + 2] === 2,
              setVisibility: setKptVisibility,
              bodypart:
                bodypartsRef.current[
                  markersTable.current[idx].length % bodypartsRef.current.length
                ],
            };
            markersTable.current[idx].push(marker);
          }
        }
        if (idx === currImgIdxRef.current)
          setMarkers(markersTable.current[idx]);
      }
    });
  }, [props.datasetName, props.imgList]);

  useEffect(() => {
    const newMarkers = markersRef.current.map((x) => {
      return { ...x, r: circleRadius };
    });
    setMarkers(newMarkers);
  }, [circleRadius, markersRef]);

  useEffect(() => {
    if (markersTable.current[currImgIdx])
      setMarkers(markersTable.current[currImgIdx]);
    else setMarkers([]);
  }, [currImgIdx, markersTable]);

  useEffect(() => {
    let colorList = [];
    for (let i = 0; i < objNum; i++) colorList.push(randomColor());
    setColors(colorList);
  }, [objNum]);

  useEffect(() => {
    if (kptVisibility.id !== undefined) {
      const newMarkers = [...markersRef.current];
      const vis = newMarkers[kptVisibility.id].visible;
      newMarkers[kptVisibility.id].visible = !vis;
      setMarkers(newMarkers);
    }
  }, [markersRef, kptVisibility]);

  useEffect(() => {
    const newMarkers = [...markersRef.current];
    newMarkers.forEach(
      (x, i) => (x.color = colors[Math.floor(i / bodypartsRef.current.length)])
    );
    setMarkers(newMarkers);
    for (let idx of Object.keys(markersTable.current)) {
      for (let i = 0; i < markersTable.current[idx].length; i++)
        markersTable.current[idx][i].color =
          colors[Math.floor(i / bodypartsRef.current.length)];
    }
  }, [colors]);

  const colorPickers = colors.map((x, i) => (
    <HexColorPicker
      color={x}
      onChange={(c) => {
        let newColors = [...colors];
        newColors[i] = c;
        setColors(newColors);
      }}
    />
  ));

  const content = (
    <Space direction="vertical" style={{ maxWidth: 400 }}>
      <Divider orientation="left">Basic</Divider>
      <Space>
        Object Number:
        <InputNumber
          min={1}
          defaultValue={objNum}
          onChange={(value) => {
            console.log(objNum);
            setObjNum(value);
          }}
        />
        Circle Radius:
        <InputNumber min={0} value={circleRadius} onChange={setCircleRadius} />
      </Space>
      <Divider orientation="left">Colors</Divider>
      <Space>
        {colors.map((c, i) => (
          <Popover content={colorPickers[i]} title="Color Panel">
            <div
              style={{
                backgroundColor: c,
                width: 36,
                height: 18,
                borderRadius: 2,
                cursor: "pointer",
                margin: 2,
              }}
            ></div>
          </Popover>
        ))}
      </Space>
      <Divider orientation="left">Bodyparts</Divider>
      <Space wrap>
        {props.bodyparts.map((x) => (
          <Tag>{x}</Tag>
        ))}
      </Space>
    </Space>
  );
  return (
    <>
      <div style={{ position: "relative" }}>
        <svg
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            zIndex: 2,
          }}
          viewBox={`0 0 ${props.imgList[currImgIdx].width} ${props.imgList[currImgIdx].height}`}
          ref={svg}
          preserveAspectRatio="none"
        >
          {markers.map((props) => (
            <Marker {...props} />
          ))}
        </svg>
        <img
          ref={imgRef}
          src={`${global.constants.staticAPI}/${props.imgList[currImgIdx].src}`}
          alt={props.imgList[currImgIdx].name}
          style={{
            // position:'absolute',
            width: "100%",
            top: 0,
            left: 0,
            height: "auto",
          }}
        />
      </div>
      <div style={{ textAlign: "center" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-around",
            margin: "5px 0px",
            alignItems: "center",
          }}
        >
          <Pagination
            defaultCurrent={currImgIdx}
            current={currImgIdx + 1}
            simple
            total={props.imgList.length}
            pageSize={1}
            onChange={(page, pageSize) => {
              markersTable.current[currImgIdx] = [...markers];
              setCurrImgIdx(page - 1);
            }}
          />
          {props.children}
          <Space>
            <Popover
              content={content}
              trigger="click"
              style={{ maxWidth: 400 }}
            >
              <Button icon={<SettingOutlined />} />
            </Popover>
            <Button
              icon={<SaveOutlined />}
              onClick={() => {
                saveAnnotations(
                  props.imgList,
                  props.bodyparts.length,
                  objNum,
                  markers,
                  currImgIdx,
                  props.datasetName
                );
              }}
            />
            <Button
              icon={<DeleteOutlined />}
              onClick={() => {
                console.log(props.imgList[currImgIdx]);
                axios
                  .post("images/delete", {
                    imageName: props.imgList[currImgIdx].name,
                  })
                  .then((res) => {
                    if (res.data.status === "success") {
                      message.success(res.data.message); // TODO: probably need to setImgList
                    } else message.error(res.data.message);
                  })
                  .catch((e) => {
                    console.log(e);
                    message.error("internal error");
                  });
              }}
            />
          </Space>
        </div>
        {currImgIdx >= props.imgList.length - 1 ? (
          <EmptyAnnotator
            plain={true}
            selectedAnimal={props.selectedAnimal}
            createMoreData={props.createMoreData}
          />
        ) : null}
      </div>
    </>
  );
};

const Marker = (props) => {
  // TODO: better svg marker
  const marker = useRef();
  const textRef = useRef();
  const [hover, setHover] = useState(false);
  const hoverRef = useRef();
  const visibleRef = useRef();
  visibleRef.current = props.visible;

  useEffect(() => {
    if (marker.current) {
      const setVisibility = (e) => {
        e.stopPropagation();
        let visibility = !visibleRef.current;
        console.log(props.id, visibility);
        props.setVisibility({ id: props.id, visible: visibility });
      };

      marker.current.onmousedown = setVisibility;
      marker.current.onmouseover = (e) => {
        setHover(true);
      };
      marker.current.onmouseout = (e) => setHover(false);
      textRef.current.onmousedown = setVisibility;
      textRef.current.onmouseover = (e) => setHover(true);
      textRef.current.onmouseout = (e) => setHover(false);
    }
  }, [marker, visibleRef, hoverRef, props.id, props.setVisibility, textRef]);

  return (
    <g>
      <circle
        cx={props.cx}
        cy={props.cy}
        r={props.r}
        ref={marker}
        fill={props.color}
        opacity={props.visible ? 1 : 0.5}
        style={{ userSelect: "none" }}
      />
      <g opacity={hover ? 1 : 0} ref={textRef}>
        {/* <text x={props.cx} y={props.cy} font-size={props.r * 1.5} fill="red">{props.bodypart}</text> */}
      </g>
    </g>
  );
};

const mapStateToProps = (state, props) => {
  console.log(state.params.cropArea.width);
  return {
    videoPath: state.originalVideo.originalPath,
    width: state.params.cropArea.width,
    height: state.params.cropArea.height,
    maxDet: state.params.maxDet,
    cropArea: state.params.cropArea,
    cropTime: state.params.cropTime,
    ...props,
  };
};

const mapDispatchToProps = null;

const VisibleAnnotator = connect(
  mapStateToProps,
  mapDispatchToProps
)(AnnotatorWrapper);

export default VisibleAnnotator;
