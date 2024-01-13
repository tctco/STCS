import { useEffect, useState, useRef } from "react";
import styles from "./VideoPlayer.module.css";
import {
  Button,
  Switch,
  Slider,
  Tooltip,
  Space,
  Modal,
  Input,
  InputNumber,
  message,
  Drawer,
  ColorPicker,
} from "antd";
import {
  PlayCircleTwoTone,
  PauseCircleTwoTone,
  DownloadOutlined,
  createFromIconfontCN,
} from "@ant-design/icons";
import * as d3 from "d3";
import ProgressBar from "./ProgressBar";
import { randomColor } from "../utils";
import userService from "../services/user.service";
import axios from "axios";
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

const IconFont = createFromIconfontCN({
  scriptUrl: "////at.alicdn.com/t/font_2723840_px795wck4bq.js",
});

export const VideoWithSVG = (props) => {
  const [playing, setPlaying] = useState(false);
  const [isVideoVisible, setVideoVisibility] = useState(true);
  const videoRef = useRef(null);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [mode, setMode] = useState("None");
  const [currFrame, setCurrFrame] = useState(0);
  const [videoTime, setVideoTime] = useState(0);
  const [settedVideoTime, setSettedVideoTime] = useState(0);
  const [displaySkeletons, setDisplaySkeletons] = useState(false);
  const [trackResult, setTrackResult] = useState(null);
  const [loadingTrackResult, setLoadingTrackResult] = useState(false);
  const [dataUrl, setDataUrl] = useState(null);
  useEffect(() => {
    setTrackResult(null);
    setDisplaySkeletons(false);
  }, [props.videoId]);

  useEffect(() => {
    const recordProgress = (_, metadata) => {
      setVideoTime(metadata.mediaTime);
      let frame = Math.round(metadata.mediaTime * props.fps);
      setCurrFrame(frame);
      if (videoRef.current) {
        videoRef.current.requestVideoFrameCallback(recordProgress);
      }
    };
    videoRef.current.requestVideoFrameCallback(recordProgress);
  }, [props.fps]);

  useEffect(() => {
    videoRef.current.currentTime = settedVideoTime;
  }, [settedVideoTime]);

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

  const handlePlaybackRateSlide = (value) => {
    videoRef.current.playbackRate = value;
    setPlaybackRate(value);
  };

  return (
    <div style={{ textAlign: "center" }}>
      {trackResult && trackResult.data && displaySkeletons ? (
        <Plotter
          trackNum={trackResult.data.length}
          mode={mode}
          setMode={setMode}
          setScale={props.setScale}
          trackData={trackResult.data}
          currFrame={currFrame}
          connections={trackResult.headers.connections}
          interval={trackResult.headers.interval}
          frameShift={0}
          behaviorData={props.behaviorData}
          width={props.width}
          height={props.height}
        />
      ) : null}

      <video
        src={props.src}
        style={{
          visibility: isVideoVisible ? "visible" : "hidden",
          width: "100%",
        }}
        ref={videoRef}
      />
      <Space>
        <Button
          type="dashed"
          shape="circle"
          icon={playing ? <PauseCircleTwoTone /> : <PlayCircleTwoTone />}
          onClick={handleButtonClick}
        />
        <span>
          {videoRef.current
            ? convertSecToISO(videoTime, true) +
              "/" +
              convertSecToISO(videoRef.current.duration, true)
            : "00:00:00/00:00:00"}
        </span>
        <Slider
          defaultValue={1}
          min={0}
          max={8}
          step={0.1}
          value={playbackRate}
          style={{ minWidth: 120 }}
          onChange={handlePlaybackRateSlide}
        />
        <Tooltip
          placement="bottom"
          title={isVideoVisible ? "Hide the Video" : "Show the Video"}
        >
          <Switch onChange={setVideoVisibility} defaultChecked size="small" />
        </Tooltip>
        <Tooltip placement="bottom" title="Display Skeletons">
          <Switch
            loading={loadingTrackResult}
            checked={displaySkeletons}
            onChange={(checked) => {
              if (checked) {
                if (trackResult === null) {
                  setLoadingTrackResult(true);
                  setDataUrl(null);
                  userService
                    .getTrackResult(props.videoId)
                    .then((res) => {
                      setDataUrl(`${res.data.path}`);
                      axios.get(`${res.data.path}`, {baseURL: '/'}).then((res) => {
                        setTrackResult(res.data);
                        setLoadingTrackResult(false);
                        setDisplaySkeletons(true);
                      });
                    })
                    .catch((e) => {
                      const msg = e.response.data.message
                        ? e.response.data.message
                        : e.message;
                      message.error(msg);
                      setLoadingTrackResult(false);
                      setDisplaySkeletons(false);
                    });
                } else setDisplaySkeletons(true);
              } else setDisplaySkeletons(false);
            }}
            size="small"
          />
        </Tooltip>
        <Button icon={<DownloadOutlined />} href={dataUrl} download />
        {/* <Button
          type="primary"
          shape="circle"
          onClick={() => setMode("SCALE")}
          loading={mode === "SCALE"}
          icon={<IconFont type="icon-ruler" />}
        /> */}
      </Space>
      <ProgressBar
        cropTime={false}
        setVideoTime={setSettedVideoTime}
        pause={pause}
        currentTime={videoTime}
        duration={videoRef.current ? videoRef.current.duration : 0}
      />
    </div>
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
  const [scaleDist, setScaleDist] = useState(0);
  const [scaleUnit, setScaleUnit] = useState("");
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
      <Modal
        title="Scale"
        centered
        visible={isScaleModalVisible}
        onOk={() => {
          setScaleModalVisibility(false);
          props.setMode("NONE");
          if (scaleDist && scaleDist > 0) {
            let ratio = scalePixel / scaleDist;
            props.setScale({ ratio: ratio, unit: scaleUnit });
          }
        }}
        onCancel={() => {
          clearSVG(scaleRef.current);
          setScaleModalVisibility(false);
          props.setMode("NONE");
        }}
      >
        <Space>
          <InputNumber
            value={scalePixel}
            disabled
            addonBefore={<span>Pixel: </span>}
          />
          <InputNumber
            value={scaleDist}
            onChange={setScaleDist}
            addonBefore={<span>Distance: </span>}
          />
          <Input
            value={scaleUnit}
            onChange={(e) => {
              setScaleUnit(e.target.value);
            }}
            addonBefore={<span>Unit: </span>}
          />
        </Space>
      </Modal>
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
