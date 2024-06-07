import React, { useEffect, useRef } from "react";
import * as echarts from "echarts";
import { CustomSeriesRenderItemReturn } from "echarts";
import { randomColor } from "../utils";
import { useVideo } from "../context/videoProvider";
import { Empty } from "antd";
import { IntervalData } from "../types";

interface LineChartComponentProps {
  data: IntervalData[];
  maxDet: number;
}

function renderItem(params, api) {
  var categoryIndex = api.value(0);
  var start = api.coord([api.value(1), categoryIndex]);
  var end = api.coord([api.value(2), categoryIndex]);
  var height = api.size([0, 1])[1] * 0.6;
  var rectShape = echarts.graphic.clipRectByRect(
    {
      x: start[0],
      y: start[1] - height / 2,
      width: end[0] - start[0],
      height: height,
    },
    {
      x: params.coordSys.x,
      y: params.coordSys.y,
      width: params.coordSys.width,
      height: params.coordSys.height,
    }
  );
  return (
    rectShape &&
    ({
      type: "rect",
      transition: ["shape"],
      shape: rectShape,
      style: api.style(),
    } as CustomSeriesRenderItemReturn)
  );
}

const TrackletsPlot: React.FC<LineChartComponentProps> = ({ data, maxDet }) => {
  const chartRef = useRef<HTMLDivElement>(null);
  let convertedData = [];
  let counter = maxDet;
  for (let i = 0; i < data.length; i++) {
    let category: number;
    if (!data[i].trackID) {
      counter += 1;
      category = counter;
    } else {
      category = data[i].trackID;
    }
    for (let j = 0; j < data[i].intervals.length; j += 2) {
      convertedData.push({
        value: [
          category - 1,
          data[i].intervals[j],
          data[i].intervals[j + 1] + 1,
        ],
        name: "Tracklet " + data[i].rawTrackID + " (" + category + ")",
        itemStyle: {
          color: randomColor((category - 1).toString()),
        },
      });
    }
  }

  useEffect(() => {
    if (chartRef.current) {
      const chartInstance: echarts.ECharts = echarts.init(chartRef.current);

      const option: echarts.EChartsOption = {
        tooltip: {
          formatter: function (params) {
            return (
              params.marker +
              params.name +
              ": " +
              (params.value[2] - params.value[1]) +
              " frames"
            );
          },
        },
        dataZoom: [
          {
            type: "slider",
            filterMode: "weakFilter",
            showDataShadow: false,
            top: 60 * maxDet,
            labelFormatter: "",
          },
          {
            type: "inside",
            filterMode: "weakFilter",
          },
        ],
        grid: {
          height: 150,
        },
        xAxis: {
          min: 0,
          scale: true,
          axisLabel: {
            formatter: function (val) {
              return Math.max(0, val) + " Frame";
            },
          },
        },
        yAxis: {
          data: Array.from({ length: counter }, (_, i) => i + 1),
        },
        series: [
          {
            type: "custom",
            renderItem: renderItem,
            itemStyle: { opacity: 0.75 },
            encode: {
              x: [1, 2],
              y: 0,
            },
            data: convertedData,
          },
        ],
      };

      chartInstance.setOption(option);

      return () => {
        chartInstance.dispose();
      };
    }
  }, [data]);

  return (
    <div
      ref={chartRef}
      style={{ width: "100%", height: "100%", minHeight: 400, minWidth: 700 }}
    ></div>
  );
};

export const VideoTracksPlot: React.FC = () => {
  const { poseTrackingData } = useVideo();
  if (!poseTrackingData) return <Empty />;
  return (
    <TrackletsPlot
      maxDet={poseTrackingData.data.length}
      data={poseTrackingData.headers.tracklets}
    />
  );
};

export default TrackletsPlot;
