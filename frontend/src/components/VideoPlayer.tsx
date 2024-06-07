import React, { useEffect, useState, useRef } from "react";
import styles from "./VideoPlayer.module.css";
import {
  Button,
  Switch,
  Tooltip,
  Space,
  Modal,
  Input,
  InputNumber,
  message,
  Drawer,
  ColorPicker,
  Empty,
  Popover,
} from "antd";
import {
  DownloadOutlined,
  ToolOutlined,
  PauseOutlined,
  PlayCircleOutlined,
} from "@ant-design/icons";
import * as d3 from "d3";
import ProgressBar from "./ProgressBar";
import { downloadFile, randomColor } from "../utils";
import userService from "../services/user.service";
import { useVideo } from "../context/videoProvider";

const convertSecToISO = (seconds, turnOnHours = false) => {
  let d = Number(seconds);
  let h = Math.floor(d / 3600);
  let m = Math.floor((d % 3600) / 60);
  let s = Math.floor((d % 3600) % 60);

  let hDisplay =
    turnOnHours || h > 0 ? h.toString().padStart(2, "0") + ":" : "";
  let mDisplay = m.toString().padStart(2, "0");
  let sDisplay = s.toString().padStart(2, "0");
  return hDisplay + mDisplay + ":" + sDisplay;
};

export const SVGROIDrawBoard = (props) => {
  const svgRef = useRef(null);

  const interactiveDraw = () => {
    let svg = svgRef.current;
    let isCompete;
    let x1, y1;
    svg.onmousedown = (e) => {
      x1 = e.offsetX;
      y1 = e.offsetY;
      isCompete = false;
    };
    svg.onmousemove = (e) => {
      let x2 = e.offsetX;
      let y2 = e.offsetY;
      let width = svg.width.animVal.value;
      let height = svg.height.animVal.value;
      if (!isCompete) {
        if (
          (Math.abs(x2 - x1) > 10 || Math.abs(y2 - y1) > 10) &&
          x2 > 0 &&
          y2 > 0
        ) {
          let cropArea = {
            x: Math.round((x1 / width) * props.videoMetaData.width),
            y: Math.round((y1 / height) * props.videoMetaData.height),
            width: Math.round(((x2 - x1) / width) * props.videoMetaData.width),
            height: Math.round(
              ((y2 - y1) / height) * props.videoMetaData.height
            ),
          };
          drawRect(cropArea);
        }
      }
    };
    svg.onmouseup = (e) => {
      let x2 = e.offsetX;
      let y2 = e.offsetY;
      let width = svg.width.animVal.value;
      let height = svg.height.animVal.value;
      let cropArea = {
        x: Math.round((x1 / width) * props.videoMetaData.width),
        y: Math.round((y1 / height) * props.videoMetaData.height),
        width: Math.round(((x2 - x1) / width) * props.videoMetaData.width),
        height: Math.round(((y2 - y1) / height) * props.videoMetaData.height),
      };
      if (cropArea.width > 10 && cropArea.height > 10) {
        drawRect(cropArea);
        props.setCropArea(cropArea);
      } else {
        props.setCropArea({
          x: 0,
          y: 0,
          width: props.videoMetaData.width,
          height: props.videoMetaData.height,
        });
        while (svg.lastChild) {
          svg.removeChild(svg.lastChild);
        }
      }
      isCompete = true;
    };
  };

  const drawRect = ({ x, y, width, height }) => {
    const svg = svgRef.current;
    while (svg.lastChild) {
      svg.removeChild(svg.lastChild);
    }
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    if (x >= 0 && y >= 0 && width >= 0 && height >= 0) {
      rect.setAttribute("x", x);
      rect.setAttribute("y", y);
      rect.setAttribute("width", width);
      rect.setAttribute("height", height);
      rect.setAttribute(
        "style",
        "fill:blue; stroke-width:1; stroke:black; fill-opacity:0.1;"
      );
      svg.appendChild(rect);
    }
  };

  useEffect(() => {
    props.setCropArea({
      x: 0,
      y: 0,
      width: props.videoMetaData.width,
      height: props.videoMetaData.height,
    });
    interactiveDraw();
  }, [props.videoMetaData, props.setCropArea]);

  return (
    <svg
      className={styles.drawBoard}
      viewBox={`0 0 ${props.videoMetaData.width} ${props.videoMetaData.height}`}
      preserveAspectRatio="none"
      ref={svgRef}
    />
  );
};

export const VideoWithSVG = (props) => {
  const [playing, setPlaying] = useState(false);
  const [isVideoVisible, setVideoVisibility] = useState(true);
  const videoRef = useRef(null);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [mode, setMode] = useState("None");
  const [displaySkeletons, setDisplaySkeletons] = useState(false);
  const [loadingTrackResult, setLoadingTrackResult] = useState(false);
  const {
    videoMeta,
    currentTime,
    setCurrentTime,
    timeSetter,
    setTimeSetter,
    poseTrackingData,
    setPoseTrackingData,
  } = useVideo();

  useEffect(() => {
    setPoseTrackingData(null);
    setDisplaySkeletons(false);
  }, [videoMeta]);

  useEffect(() => {
    if (!videoRef.current) return;
    const recordProgress = (_, metadata) => {
      let frame = Math.round(metadata.mediaTime * videoMeta.fps);
      setTimeSetter("video");
      setCurrentTime(frame);
      if (videoRef.current) {
        videoRef.current.requestVideoFrameCallback(recordProgress);
      }
    };
    videoRef.current.requestVideoFrameCallback(recordProgress);
  }, [videoMeta, setCurrentTime, setTimeSetter, videoRef]);

  useEffect(() => {
    if (timeSetter !== "video") {
      videoRef.current.currentTime = currentTime / videoMeta.fps;
    }
  }, [videoRef, currentTime, timeSetter, videoMeta]);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.addEventListener("ended", () => {
        setPlaying(false);
      });
    }
  }, [videoRef.current]);

  const handleButtonClick = () => {
    if (videoRef.current.paused || videoRef.current.ended) play();
    else pause();
  };

  const play = () => {
    setPlaying(true);
    videoRef.current.play();
  };

  const pause = () => {
    setPlaying(false);
    videoRef.current.pause();
  };

  return videoMeta ? (
    <div style={{ textAlign: "center" }}>
      {poseTrackingData && poseTrackingData.data && displaySkeletons ? (
        <Plotter
          trackNum={poseTrackingData.data.length}
          mode={mode}
          setMode={setMode}
          setScale={props.setScale}
          trackData={poseTrackingData.data}
          currFrame={Math.round(currentTime)}
          connections={poseTrackingData.headers.connections}
          interval={poseTrackingData.headers.interval}
          frameShift={0}
          behaviorData={poseTrackingData}
          width={videoMeta.width}
          height={videoMeta.height}
        />
      ) : null}

      <video
        src={`http://localhost${videoMeta.url}`}
        style={{
          visibility: isVideoVisible ? "visible" : "hidden",
          width: "100%",
        }}
        ref={videoRef}
      />
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          width: "100%",
        }}
      >
        <Button
          type="dashed"
          icon={playing ? <PauseOutlined /> : <PlayCircleOutlined />}
          onClick={handleButtonClick}
        />
        <div style={{ flexGrow: 1, marginLeft: 5, marginRight: 5 }}>
          <ProgressBar
            cropTime={false}
            setVideoTime={setCurrentTime}
            setTimeSetter={setTimeSetter}
            pause={pause}
            currentTime={currentTime / videoMeta.fps}
            duration={videoRef.current ? videoRef.current.duration : 0}
            fps={videoMeta ? videoMeta.fps : 1}
          />
        </div>
        <span style={{ marginRight: 5, fontSize: "0.75em" }}>
          {videoRef.current
            ? convertSecToISO(currentTime / videoMeta.fps, true) +
              "/" +
              convertSecToISO(videoRef.current.duration, true)
            : "00:00:00/00:00:00"}
        </span>
        <Popover
          title="Toolbox"
          trigger="click"
          content={
            <Space>
              <InputNumber
                // @ts-ignore
                changeOnWheel
                addonBefore="Playback Rate"
                defaultValue={1}
                min={0}
                max={100}
                step={0.1}
                value={playbackRate}
                size="small"
                onChange={(v) => {
                  videoRef.current.playbackRate = v;
                  setPlaybackRate(v);
                }}
              />
              <Tooltip
                title={isVideoVisible ? "Hide the Video" : "Show the Video"}
              >
                <Switch
                  onChange={setVideoVisibility}
                  defaultChecked
                  size="small"
                />
              </Tooltip>
              <Tooltip title="Display Skeletons">
                <Switch
                  loading={loadingTrackResult}
                  checked={displaySkeletons}
                  onChange={(checked) => {
                    if (checked) {
                      console.log("prev catch", poseTrackingData);
                      if (!poseTrackingData) {
                        setLoadingTrackResult(true);
                        console.log("catch", videoMeta.id);
                        userService
                          .getTrackResult(videoMeta.id)
                          .then((res) => {
                            console.log("catch", res.data);
                            setPoseTrackingData(res.data);
                            setLoadingTrackResult(false);
                            setDisplaySkeletons(true);
                          })
                          .catch((e) => {
                            const msg = e.response.data.message
                              ? e.response.data.message
                              : e.message;
                            message.error(msg);
                            console.log(e);
                            setLoadingTrackResult(false);
                            setDisplaySkeletons(false);
                          });
                      } else {
                        setDisplaySkeletons(true);
                        console.log(poseTrackingData);
                      }
                    } else setDisplaySkeletons(false);
                  }}
                  size="small"
                />
              </Tooltip>
              <Tooltip title="Download the Track Result">
                <Button
                  icon={<DownloadOutlined />}
                  loading={loadingTrackResult}
                  onClick={() => {
                    if (poseTrackingData) {
                      downloadFile(poseTrackingData, videoMeta.name + ".json");
                      return;
                    } else if (!videoMeta.analyzed) {
                      message.error("No track result available");
                      return;
                    } else if (!poseTrackingData) {
                      setLoadingTrackResult(true);
                      userService
                        .getTrackResult(videoMeta.id)
                        .then((res) => {
                          setLoadingTrackResult(false);
                          setPoseTrackingData(res.data);
                          downloadFile(res.data, videoMeta.name + ".json");
                        })
                        .catch((e) => {
                          message.error(e.response.data.message);
                          setLoadingTrackResult(false);
                          return;
                        });
                    }
                  }}
                />
              </Tooltip>
              {/* <Button
          type="primary"
          shape="circle"
          onClick={() => setMode("SCALE")}
          loading={mode === "SCALE"}
          icon={<IconFont type="icon-ruler" />}
        /> */}
            </Space>
          }
        >
          <Tooltip title="Toolbox">
            <Button icon={<ToolOutlined />}></Button>
          </Tooltip>
        </Popover>
      </div>
    </div>
  ) : (
    <Empty
      imageStyle={{ height: 250 }}
      description="Select an uploaded video to start"
    />
  );
};

const clearSVG = (svg) => {
  while (svg.lastChild) svg.removeChild(svg.lastChild);
};

const drawLine = (x1, y1, x2, y2) => {
  let line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1);
  line.setAttribute("x2", x2);
  line.setAttribute("y1", y1);
  line.setAttribute("y2", y2);
  return line;
};

const getSVGCoords = (svg, e) => {
  let pt = svg.createSVGPoint();
  pt.x = e.clientX;
  pt.y = e.clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
};

const Plotter = (props) => {
  const svgRef = useRef(null);
  const tracksRef = useRef(null);
  const tracksEntriesRef = useRef([]);
  const [isScaleModalVisible, setScaleModalVisibility] = useState(false);
  const [scalePixel, setScalePixel] = useState(-1);
  const [openDrawer, setOpenDrawer] = useState(false);
  const scaleRef = useRef(null);
  const initTracksProps = props.trackData.map((_, i) => ({
    color: randomColor(i.toString()),
    visibility: true,
  }));
  const [tracksProps, setTracksProps] = useState(initTracksProps);
  useEffect(() => {
    if (!props.trackData) return;
    let newProps = [];
    for (let i = 0; i < props.trackData.length; i++) {
      newProps.push({
        color: randomColor(i.toString()),
      });
    }
    setTracksProps(newProps);
  }, [props.trackData]);

  useEffect(() => {
    if (!tracksRef.current) return;
    clearSVG(tracksRef.current);
    tracksEntriesRef.current = new Array(props.trackNum);
    for (let track = 0; track < props.trackNum; track++) {
      let trackGroup = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "g"
      );
      tracksRef.current.appendChild(trackGroup);
      tracksEntriesRef.current[track] = trackGroup;
    }
  }, [tracksRef.current, props.trackNum]);

  useEffect(() => {
    if (props.mode === "SCALE" && svgRef.current && scaleRef.current) {
      let x1, y1, x2, y2;
      const handleMouseDown = (e) => {
        let loc = getSVGCoords(svgRef.current, e);
        x1 = loc.x;
        y1 = loc.y;
        svgRef.current.addEventListener("mousemove", handleMouseMove);
        svgRef.current.addEventListener("mouseup", handleMouseUp);
      };
      const handleMouseMove = (e) => {
        let loc = getSVGCoords(svgRef.current, e);
        x2 = loc.x;
        y2 = loc.y;
        if (Math.abs(x1 - x2) > 3 || Math.abs(y1 - y2) > 3) {
          clearSVG(scaleRef.current);
          let line = drawLine(x1, y1, x2, y2);
          line.setAttribute("style", "stroke-width: 1px; stroke: magenta");
          scaleRef.current.appendChild(line);
        }
      };
      const handleMouseUp = () => {
        let dist = Math.round(Math.hypot(x2 - x1, y2 - y1));
        setScalePixel(dist);
        setScaleModalVisibility(true);
        svgRef.current.removeEventListener("mousedown", handleMouseDown);
        svgRef.current.removeEventListener("mousemove", handleMouseMove);
        svgRef.current.removeEventListener("mouseup", handleMouseUp);
      };

      svgRef.current.addEventListener("mousedown", handleMouseDown);
    }
  }, [props.mode, svgRef.current, scaleRef.current]);

  useEffect(() => {
    if (
      !props.trackData ||
      !props.currFrame ||
      props.currFrame + props.frameShift >= props.interval[1] ||
      props.currFrame + props.frameShift < props.interval[0]
    )
      return;
    let currFrame = props.currFrame - props.interval[0] + props.frameShift;
    for (let track = 0; track < props.trackNum; track++) {
      let svg = d3.select(tracksEntriesRef.current[track]);
      svg
        .selectAll("line")
        .each(function () {
          this.bogus_opacity *= 0.9;
        })
        .attr("stroke-opacity", function () {
          return this.bogus_opacity;
        })
        .attr("stroke", tracksProps[track].color)
        .filter(function () {
          return this.bogus_opacity < 0.05;
        })
        .remove();
      svg
        .selectAll(null)
        .data(props.connections)
        .enter()
        .append("line")
        .each(function () {
          this.bogus_opacity = 1.0;
        })
        .attr("x1", (d) => props.trackData[track][currFrame][d[0]][0])
        .attr("y1", (d) => props.trackData[track][currFrame][d[0]][1])
        .attr("x2", (d) => props.trackData[track][currFrame][d[1]][0])
        .attr("y2", (d) => props.trackData[track][currFrame][d[1]][1])
        .attr("visibility", (d) => {
          if (
            props.trackData[track][currFrame][d[0]][0] <= 0 ||
            props.trackData[track][currFrame][d[1]][0] <= 0 ||
            tracksProps[track].visibility === false
          ) {
            return "hidden";
          }
        })
        .attr("stroke", tracksProps[track].color)
        .attr("stroke-width", 1);
    }
  }, [
    props.currFrame,
    props.trackData,
    props.trackNum,
    props.connections,
    props.interval,
    props.behaviorData,
    tracksEntriesRef,
    tracksProps,
  ]);

  return (
    <>
      <svg
        className={styles.drawBoard}
        viewBox={`0 0 ${props.width} ${props.height}`}
        preserveAspectRatio="none"
        ref={svgRef}
        onClick={() => setOpenDrawer(true)}
      >
        <g ref={tracksRef} />
        <g ref={scaleRef} />
      </svg>
      <ScaleModal
        open={isScaleModalVisible}
        setOpen={setScaleModalVisibility}
        pixelLength={scalePixel}
        svgRef={svgRef}
      />
      <Drawer
        title="Tracks"
        getContainer={false}
        open={openDrawer}
        width={"30%"}
        closable={false}
        onClose={() => setOpenDrawer(false)}
      >
        <Space wrap={true}>
          {tracksProps.map((trackProp, i) => (
            <Space>
              <ColorPicker
                value={trackProp.color}
                onChangeComplete={(v) =>
                  setTracksProps((prev) => {
                    let newProps = [...prev];
                    newProps[i].color = v.toHexString();
                    return newProps;
                  })
                }
              />
              <Switch
                defaultChecked={true}
                checked={trackProp.visibility}
                onChange={(v) =>
                  setTracksProps((prev) => {
                    console.log(v);
                    let newProps = [...prev];
                    newProps[i].visibility = v;
                    return newProps;
                  })
                }
              />
            </Space>
          ))}
        </Space>
      </Drawer>
    </>
  );
};

interface ScaleModalProps {
  open: boolean;
  setOpen: (visible: boolean) => void;
  pixelLength: number;
  svgRef: React.RefObject<SVGSVGElement>;
}

const ScaleModal: React.FC<ScaleModalProps> = ({
  open,
  setOpen,
  pixelLength,
  svgRef,
}) => {
  const { scale, setScale } = useVideo();
  const [realWorldDist, setRealWorldDist] = useState(pixelLength / scale.ratio);
  const [unit, setUnit] = useState("");
  return (
    <Modal
      title="Scale"
      centered
      open={open}
      onOk={() => {
        setOpen(false);
        // props.setMode("NONE");
        if (pixelLength > 0 && realWorldDist > 0) {
          let ratio = pixelLength / realWorldDist;
          setScale({ ratio, unit });
        }
      }}
      onCancel={() => {
        clearSVG(svgRef.current);
        setOpen(false);
        // props.setMode("NONE");
      }}
    >
      <Space>
        <InputNumber
          value={pixelLength}
          disabled
          addonBefore={<span>Pixel: </span>}
        />
        <InputNumber
          value={realWorldDist}
          min={0}
          onChange={(value) => {
            if (value > 0 && typeof value === "number") setRealWorldDist(value);
          }}
          addonBefore={<span>Distance: </span>}
        />
        <Input
          value={unit}
          onChange={(e) => {
            setUnit(e.target.value);
          }}
          addonBefore={<span>Unit: </span>}
        />
      </Space>
    </Modal>
  );
};
