import React, { useState, useEffect, useRef } from "react";
import { useParams } from "react-router-dom";
import type { UploadProps } from "antd";
import {
  Upload,
  message,
  Button,
  Tooltip,
  Empty,
  Space,
  Pagination,
  Skeleton,
  Select,
  Form,
  Divider,
  ColorPicker,
  InputNumber,
} from "antd";
import {
  FileImageOutlined,
  FileZipOutlined,
  PlusOutlined,
} from "@ant-design/icons";
import userService from "../services/user.service";
import authHeader from "../services/auth-header";
import { useDataset } from "../context/datasetProvider";
import { logError } from "../utils";
import paper, { Tool, Path, Point, Size, Rectangle } from "paper";
import { randomColor } from "../utils";
import type { Annotation } from "../types";

type ImageInfo = {
  id: number;
  url: string;
  difficult: boolean;
  annotations: Annotation[];
};

const uploadZipProps: UploadProps = {
  beforeUpload: (file) => {
    const isZip =
      file.type === "application/zip" ||
      file.type === "application/x-zip-compressed";
    if (!isZip) {
      message.error(`${file.name} is not a zip file`);
    }
    const isLt2G = file.size / 1024 / 1024 / 1024 < 2;
    if (!isLt2G) {
      message.error("Image must smaller than 2GB!");
    }
    return isZip && isLt2G;
  },
  onChange: (info) => {
    if (info.file.response && !info.file.response.success)
      logError(info.file.response);
  },
  maxCount: 1,
  headers: authHeader(),
};

const AnnotationPage: React.FC = () => {
  const { datasetId } = useParams();
  const [currentImage, setCurrentImage] = useState(0);
  const { dataset, setDataset } = useDataset();
  const [imageList, setImageList] = useState<ImageInfo[]>([]);
  const [loadingImageList, setLoadingImageList] = useState(false);

  const uploadImageListProps: UploadProps = {
    multiple: true,
    accept: "image/*",
    action: userService.getUploadImageApi(parseInt(datasetId)),
    headers: authHeader(),
  };

  useEffect(() => {
    if (dataset) return;
    userService
      .getDatasetById(parseInt(datasetId))
      .then((response) => {
        setDataset(response.data);
      })
      .catch((e) => {
        logError(e);
      });
  }, [dataset, datasetId]);

  useEffect(() => {
    if (datasetId) {
      setImageList([]);
      setLoadingImageList(true);
      userService
        .getImageList(parseInt(datasetId))
        .then((response) => {
          console.log(response.data);
          setImageList(response.data);
          setLoadingImageList(false);
        })
        .catch((e) => {
          logError(e);
          setLoadingImageList(false);
        });
    }
  }, [datasetId]);

  return (
    <>
      {loadingImageList ? (
        <Skeleton />
      ) : (
        <Space direction="vertical" style={{ textAlign: "center" }}>
          <Pagination
            defaultCurrent={0}
            pageSize={1}
            total={imageList.length + 1}
            onChange={(page) => {
              setCurrentImage(page - 1);
            }}
          />
          {currentImage < imageList.length && dataset ? (
            <AnnotationTool
              imageUrl={`http://localhost${imageList[currentImage].url}`}
              keypoints={dataset.keypoints}
              imageId={imageList[currentImage].id}
            />
          ) : (
            <Empty>
              <Space>
                <Upload {...uploadImageListProps}>
                  <Button icon={<FileImageOutlined />}>Upload Images</Button>
                </Upload>
                <Tooltip title="Wrap data.json and images in a .zip file to upload">
                  <Upload
                    {...uploadZipProps}
                    action={
                      dataset ? userService.getUploadCOCOApi(dataset.id) : ""
                    }
                  >
                    <Button icon={<FileZipOutlined />}>Upload COCO Zip</Button>
                  </Upload>
                </Tooltip>
              </Space>
            </Empty>
          )}
        </Space>
      )}
    </>
  );
};

const allZeros = (arr: number[]): boolean => {
  return arr.every((v) => v === 0);
};

interface AnnotationToolProps {
  imageUrl: string;
  keypoints: string[];
  imageId: number;
}

const createRectPoint = (
  point: paper.Point,
  color: string,
  size: number = 6
): paper.Item => {
  const pointRect = new paper.Path.Rectangle(
    new Rectangle(
      point.subtract(new Point(size / 2, size / 2)),
      new Size(size, size)
    )
  );
  pointRect.fillColor = new paper.Color(color);
  pointRect.opacity = 1;
  return pointRect;
};

const createKeypoint = (
  point: paper.Point,
  color: string,
  size: number = 6
) => {
  const pointRect = createRectPoint(point, color, size);
  pointRect.onMouseDown = function (e: paper.ToolEvent) {
    if ((e as any).event.button === 2) {
      if (this.opacity === 1) this.opacity = 0.5;
      else this.opacity = 1;
    }
  };
  pointRect.onMouseDrag = (e: paper.ToolEvent) => {
    if (pointRect.selected) pointRect.position = e.point;
  };
  pointRect.onMouseUp = (_e: paper.ToolEvent) => {
    pointRect.selected = false;
  };
  return pointRect;
};

const adjustRectSize = (rect: paper.Item, newSize: number) => {
  const currentSize = rect.bounds.size;
  const scaleX = newSize / currentSize.width;
  const scaleY = newSize / currentSize.height;

  rect.scale(scaleX, scaleY);
};

const createPointHitTest = (point: paper.Point, tolerance: number) => {
  const hitResult = paper.project.hitTest(point, {
    class: Path,
    fill: true,
    tolerance: tolerance,
  });
  return hitResult;
};

const convertArrayToSegments = (pointsArray: number[]): paper.Point[] => {
  let segments: paper.Point[] = [];
  for (let i = 0; i < pointsArray.length; i += 2) {
    segments.push(new paper.Point(pointsArray[i], pointsArray[i + 1]));
  }
  return segments;
};

const convertArrayToKeypoints = (
  pointsArray: number[],
  color: string,
  size: number = 6
): paper.Item[] => {
  let keypoints: paper.Item[] = [];
  for (let i = 0; i < pointsArray.length; i += 3) {
    const point = new paper.Point(pointsArray[i], pointsArray[i + 1]);
    const pointRect = createKeypoint(point, color, size);
    if (pointsArray[i + 2] === 0 || pointsArray[i + 2] === 1)
      pointRect.opacity = 0.5;
    keypoints.push(pointRect);
  }
  return keypoints;
};

const deleteInstance = (instance: Instance) => {
  instance.paths.forEach((path) => {
    if (path && path.remove) {
      path.remove();
    }
  });

  instance.pathPoints.forEach((pointArray) => {
    pointArray.forEach((point) => {
      if (point && point.remove) {
        point.remove();
      }
    });
  });
  instance.keypoints.forEach((keypoint) => {
    if (keypoint && keypoint.remove) {
      keypoint.remove();
    }
  });
};

const convertPathToPointsArray = (path: paper.Path): number[] => {
  const pointsArray = [];
  if (path && path.segments) {
    path.segments.forEach((segment) => {
      pointsArray.push(segment.point.x, segment.point.y);
    });
  }
  return pointsArray;
};

const convertPointsToCOCOKeypoints = (
  points: paper.Item[],
  keypointLength: number
): number[] => {
  let keypoints: number[] = [];
  points.forEach((point) => {
    keypoints.push(
      point.position.x,
      point.position.y,
      point.opacity === 1 ? 2 : 1
    );
  });
  if (keypoints.length < keypointLength * 3) {
    for (let i = keypoints.length; i <= keypointLength * 3; i++) {
      keypoints.push(0);
    }
  }
  return keypoints;
};

const convertInstanceToAnnotation = (
  instance: Instance,
  keypointLength: number
) => {
  const polygon = instance.paths.map((path) => convertPathToPointsArray(path));
  const keypoints = convertPointsToCOCOKeypoints(
    instance.keypoints,
    keypointLength
  );
  return {
    id: instance.annotation.id,
    keypoints: keypoints,
    polygon: polygon,
  };
};

type Instance = {
  annotation: Annotation;
  color: string;
  paths: paper.Path[];
  pathPoints: paper.Item[][];
  keypoints: paper.Item[];
};

const AnnotationTool: React.FC<AnnotationToolProps> = ({
  imageUrl,
  keypoints,
  imageId,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const activePathRef = useRef<paper.Path | null>(null);
  const activePointsRef = useRef<paper.Item[]>([]);
  const activeTempPathRef = useRef<paper.Path | null>(null);
  const [activeTool, setActiveTool] = useState("polygon");
  const [instances, setInstances] = useState<Instance[]>([]);
  const selectedInstanceRef = useRef<Instance | null>(null);
  const [selectedInstanceIdx, setSelectedInstanceIdx] = useState<number | null>(
    null
  );
  const [colorPickerColor, setColorPickerColor] = useState<string | null>(null);
  const [canvasCursor, setCanvasCursor] = useState<string>("crosshair");
  const [markerSize, setMarkerSize] = useState<number>(6);
  const { dataset } = useDataset();

  const submitAnnotations = (
    imageId: number,
    instances: Instance[],
    keypoints: string[]
  ) => {
    const annotations = instances.map((instance) => {
      return convertInstanceToAnnotation(instance, keypoints.length);
    });
    console.log(annotations);
    userService
      .postAnnotations(dataset.id, imageId, annotations)
      .then((response) => {
        message.success(response.data.message);
      })
      .catch(logError);
  };

  useEffect(() => {
    if (!canvasRef.current) return;
    paper.setup(canvasRef.current);
    canvasRef.current.oncontextmenu = (e) => e.preventDefault();

    let raster = new paper.Raster(imageUrl);
    raster.onLoad = () => {
      paper.view.viewSize = new paper.Size(raster.width, raster.height);
      raster.position = paper.view.center;
    };

    return () => {
      paper.view.remove();
    };
  }, [imageUrl]);

  useEffect(() => {
    if (!canvasRef.current || !selectedInstanceRef.current) return;
    const paperColor = new paper.Color(selectedInstanceRef.current.color);
    selectedInstanceRef.current.paths.forEach((path) => {
      path.strokeColor = paperColor;
    });
    selectedInstanceRef.current.keypoints.forEach((point) => {
      point.fillColor = paperColor;
    });
  }, [colorPickerColor]);

  useEffect(() => {
    if (!canvasRef.current || !selectedInstanceRef.current) return;
    console.log("before", instances[0].keypoints[0].bounds.size);
    instances.forEach((instance) => {
      instance.keypoints.forEach((point) => {
        adjustRectSize(point, markerSize);
      });
    });
    console.log("after", instances[0].keypoints[0].bounds.size);
    selectedInstanceRef.current.pathPoints.forEach((points) => {
      points.forEach((point) => {
        adjustRectSize(point, markerSize);
      });
    });
  }, [markerSize, instances]);

  useEffect(() => {
    if (!canvasRef.current) return;
    userService
      .getAnnotationsByImageId(dataset.id, imageId)
      .then((response) => {
        let annotations = response.data.annotations as Annotation[];
        let newInstances = annotations.map((ann, index) => {
          const color = randomColor(index);
          const paths = ann.polygon.map(
            (p) =>
              new Path({
                segments: convertArrayToSegments(p),
                strokeColor: color,
                strokeWidth: 2,
                closed: true,
              })
          );
          const pathPoints = ann.polygon.map((p) => {
            let points = convertArrayToSegments(p);
            return points.map((point) => {
              const pointRect = createRectPoint(point, "white", markerSize);
              pointRect.visible = false;
              return pointRect;
            });
          });
          let keypoints: paper.Item[];
          if (allZeros(ann.keypoints)) keypoints = [];
          else
            keypoints = convertArrayToKeypoints(
              ann.keypoints,
              color,
              markerSize
            );
          return {
            annotation: ann,
            color: color,
            paths: paths,
            pathPoints: pathPoints,
            keypoints: keypoints,
          };
        });
        setInstances(newInstances);
      })
      .catch(logError);
  }, [imageUrl, markerSize]);

  useEffect(() => {
    let editMode: Boolean = false;
    let activePoint: paper.Item | null = null;

    const polygonTool = new Tool();

    polygonTool.onMouseDown = (e: paper.ToolEvent) => {
      if ((e as any).event.button !== 0) return;
      if (selectedInstanceRef.current === null) {
        message.error("Please select an instance first", 5);
        return;
      }
      // Left mouse button
      // check if hitted an existing point
      const hitResult = createPointHitTest(e.point, 7);

      if (hitResult && hitResult.item) {
        hitResult.item.bringToFront();
        if (
          activePathRef.current &&
          activePathRef.current.firstSegment &&
          activePathRef.current.segments[0].point.equals(
            hitResult.item.bounds.center
          ) &&
          activePathRef.current.segments.length > 2 &&
          editMode === false
        ) {
          // close path when the first point is hitted
          console.log("close path");
          activePathRef.current.closed = true;
          activePathRef.current.fullySelected = false;
          selectedInstanceRef.current.paths.push(activePathRef.current);
          selectedInstanceRef.current.pathPoints.push(activePointsRef.current);
          activePathRef.current = null; // reset
          activePointsRef.current = [];
        } else if (activePathRef.current === null) {
          // if not drwaing, activate the hitted point and path
          for (let i = 0; i < selectedInstanceRef.current.paths.length; i++) {
            const index = selectedInstanceRef.current.pathPoints[i].indexOf(
              hitResult.item
            );
            if (index !== -1) {
              activePoint = selectedInstanceRef.current.pathPoints[i][index];
              activePathRef.current = selectedInstanceRef.current.paths[i];
              activePathRef.current.fullySelected = true;
              activePointsRef.current =
                selectedInstanceRef.current.pathPoints[i];
              editMode = true;
              break;
            }
          }
        }
      } else {
        // check if hitted an existing line
        const lineHitResult = paper.project.hitTest(e.point, {
          segments: false, // 不检测段点
          stroke: true, // 检测路径的边界线
          fill: false, // 不检测填充区域
          tolerance: 5, // 点击精度容忍度
        });
        if (lineHitResult && lineHitResult.type === "stroke") {
          // if hitted a line, insert a point
          console.log(lineHitResult);
          // const insertIndex = lineHitResult.location.index + 1;
          // path?.insert(insertIndex, event.point);
          // const pointRect = createRectPoint(event.point);
          // points.splice(insertIndex, 0, pointRect);
        } else {
          // if not hit any existing points/lines, create a new point
          if (!activePathRef.current) {
            // create a new path
            editMode = false;
            activePathRef.current = new Path({
              strokeColor: selectedInstanceRef.current.color,
              strokeWidth: 2,
              fullySelected: true,
            });
          }

          const pointRect = createRectPoint(e.point, "white", markerSize);
          activePointsRef.current.push(pointRect);
          activePathRef.current.add(e.point);
        }
      }
      if (activeTempPathRef.current) {
        activeTempPathRef.current.remove();
        activeTempPathRef.current = null;
      }
    };

    polygonTool.onMouseMove = (event: paper.ToolEvent) => {
      if (
        activePathRef.current &&
        activePathRef.current.segments.length > 0 &&
        editMode === false
      ) {
        // create a preview line, which only exists when creating new points
        if (!activeTempPathRef.current) {
          activeTempPathRef.current = new paper.Path({
            segments: [activePathRef.current.lastSegment.point, event.point],
            strokeColor: selectedInstanceRef.current.color,
            strokeWidth: 2,
            dashArray: [4, 4],
          });
        } else {
          // update last point of the preview line to track the mouse
          activeTempPathRef.current.lastSegment.point = event.point;
        }
      }
    };

    polygonTool.onMouseUp = (_event: paper.ToolEvent) => {
      if (activePoint) {
        // release active point
        activePoint = null;
        activePathRef.current.fullySelected = false;
        activePathRef.current = null;
        activePointsRef.current = [];
      }
    };

    polygonTool.onMouseDrag = (event: paper.ToolEvent) => {
      if (activePoint) {
        activePoint.position = event.point;
        const index = activePointsRef.current.indexOf(activePoint);
        if (index !== -1 && activePathRef.current) {
          activePathRef.current.segments[index].point = event.point;
        }
      }
    };

    const keypointsTool = new Tool();
    keypointsTool.onMouseDown = (e: paper.ToolEvent) => {
      if (selectedInstanceRef.current === null) {
        message.error("Please select an instance first", 5);
        return;
      }
      if ((e as any).event.button === 0) {
        // Left mouse button
        const hitResult = createPointHitTest(e.point, 10);
        if (hitResult && hitResult.item) {
          const index = selectedInstanceRef.current.keypoints.indexOf(
            hitResult.item
          );
          console.log(index);
          if (index !== -1) {
            hitResult.item.bringToFront();
            hitResult.item.selected = true;
            return;
          }
        }
        if (selectedInstanceRef.current.keypoints.length > keypoints.length) {
          message.warning("Too many keypoints", 5);
          return;
        }
        const pointRect = createKeypoint(
          e.point,
          selectedInstanceRef.current.color,
          markerSize
        );
        selectedInstanceRef.current.keypoints.push(pointRect);
        pointRect.selected = true;
      }
    };

    const handlePolygonKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        // delete path when hitting esc
        if (activePathRef.current) {
          activePathRef.current.remove();
          activePathRef.current = null;
          activePointsRef.current.forEach((point) => point.remove());
          activePointsRef.current = [];
          if (activeTempPathRef.current) {
            activeTempPathRef.current.remove();
            activeTempPathRef.current = null;
          }
        }
      }
    };

    const handleKptKeydown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        selectedInstanceRef.current.keypoints.forEach((point) =>
          point.remove()
        );
        selectedInstanceRef.current.keypoints = [];
      }
    };

    if (activeTool === "polygon") {
      polygonTool.activate();
      if (selectedInstanceRef.current) {
        selectedInstanceRef.current.pathPoints.forEach((points) => {
          points.forEach((p) => (p.visible = true));
        });
      }
      window.addEventListener("keydown", handlePolygonKeydown);
      window.removeEventListener("keydown", handleKptKeydown);
      setCanvasCursor("crosshair");
    } else if (activeTool === "keypoints") {
      keypointsTool.activate();
      setCanvasCursor("crosshair");
      if (selectedInstanceRef.current) {
        selectedInstanceRef.current.pathPoints.forEach((points) => {
          points.forEach((p) => (p.visible = false));
        });
      }
      window.addEventListener("keydown", handleKptKeydown);
      window.removeEventListener("keydown", handlePolygonKeydown);
    } else if (activeTool === "panzoom") {
      paper.view.zoom = 1;
      paper.view.center = paper.view.center;
      setCanvasCursor("grab");
      window.removeEventListener("keydown", handleKptKeydown);
      window.removeEventListener("keydown", handlePolygonKeydown);
    }
    return () => {
      window.removeEventListener("keydown", handleKptKeydown);
      window.removeEventListener("keydown", handlePolygonKeydown);
    };
  }, [activeTool, markerSize]);

  const addInstance = (index: number) => {
    userService
      .putNewInstance(dataset.id, imageId)
      .then((response) => {
        message.success("New instance added");
        let newInstance: Instance = {
          annotation: {
            id: response.data.id,
            image_id: imageId,
            keypoints: [],
            area: 0,
            polygon: [],
          },
          color: randomColor(index),
          paths: [],
          pathPoints: [],
          keypoints: [],
        };
        setInstances([...instances, newInstance]);
      })
      .catch(logError);
  };

  const selectInstance = (
    instance: Instance | null,
    instanceIdx: number | null
  ) => {
    if (selectedInstanceRef.current)
      selectedInstanceRef.current.pathPoints.forEach((points) => {
        points.forEach((point) => (point.visible = false));
      });
    selectedInstanceRef.current = instance;
    if (instance === null) {
      setSelectedInstanceIdx(null);
      return;
    }
    selectedInstanceRef.current.pathPoints.forEach((points) => {
      points.forEach((point) => (point.visible = true));
    });
    setSelectedInstanceIdx(instanceIdx);
    setColorPickerColor(instance.color);
  };

  return (
    <Space direction="vertical">
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", cursor: canvasCursor }}
      />
      <Form>
        <Space>
          <Form.Item label="ToolBox">
            <Tooltip title="Select and press Esc to delete">
              <Select
                options={[
                  { value: "polygon", label: "Polygon" },
                  { value: "keypoints", label: "Keypoints" },
                  // { value: "panzoom", label: "PanZoom" },
                ]}
                style={{ minWidth: 120 }}
                value={activeTool}
                onChange={setActiveTool}
              />
            </Tooltip>
          </Form.Item>
          <Form.Item label="Instance">
            <Select
              style={{ width: 200 }}
              optionRender={(option) => {
                return (
                  <Space
                    style={{ display: "flex", justifyContent: "space-between" }}
                  >
                    <span>{option.label}</span>
                    <Button
                      danger
                      type="link"
                      onClick={() => {
                        let selectedInstance;
                        for (let i = 0; i < instances.length; i++) {
                          if (instances[i] === instances[option.value]) {
                            selectedInstance = instances[i];
                            break;
                          }
                        }
                        userService
                          .deleteInstance(
                            dataset.id,
                            imageId,
                            selectedInstance.annotation.id
                          )
                          .then((_response) => {
                            message.success("successfully deleted instance");
                            deleteInstance(selectedInstance);
                            setInstances(
                              instances.filter(
                                (inst) => inst !== instances[option.value]
                              )
                            );
                            selectInstance(null, null);
                          })
                          .catch(logError);
                      }}
                    >
                      delete
                    </Button>
                  </Space>
                );
              }}
              value={selectedInstanceIdx}
              onChange={(v) => selectInstance(instances[v], v)}
              placeholder="select an instance to start"
              dropdownRender={(menu) => (
                <>
                  {menu}
                  <Divider style={{ margin: "8px 0" }} />
                  <Button
                    type="text"
                    icon={<PlusOutlined />}
                    onClick={() => addInstance(instances.length + 1)}
                    block
                  >
                    Add Instance
                  </Button>
                </>
              )}
              options={instances.map((_inst, index) => ({
                value: index,
                label: `Instance ${index + 1}`,
              }))}
            />
          </Form.Item>
          <Form.Item>
            <ColorPicker
              disabled={selectedInstanceIdx === null}
              value={colorPickerColor}
              onChange={(c) => {
                setColorPickerColor(c.toHexString());
                selectedInstanceRef.current.color = c.toHexString();
              }}
            />
          </Form.Item>
          <Form.Item>
            <InputNumber
              min={1}
              max={100}
              step={1}
              value={markerSize}
              onChange={(v) => setMarkerSize(v)}
              disabled={selectedInstanceIdx === null}
            />
          </Form.Item>
          <Form.Item>
            <Button
              onClick={() => submitAnnotations(imageId, instances, keypoints)}
            >
              Submit
            </Button>
          </Form.Item>
        </Space>
      </Form>
    </Space>
  );
};

export default AnnotationPage;
