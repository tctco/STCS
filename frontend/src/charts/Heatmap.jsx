import React from "react";
import { connect } from "react-redux";
import * as echarts from "echarts";
import * as d3 from "d3";

const HeatMap = (props) => {
  const container = React.useRef(null);
  const svg = React.useRef(null);
  const cache = React.useRef({
    createAxis: null,
    createChart: null,
    chart: null,
  });
  const [scale, setScale] = React.useState({ ratio: 1, unit: "" });

  React.useEffect(() => {
    cache.current.createAxis = (scale) => {
      while (svg.current.lastChild) {
        svg.current.removeChild(svg.current.lastChild);
      }
      let s = d3
        .select(svg.current)
        .attr("width", props.width + 10)
        .attr("height", props.height + 10);
      let width = Math.round(props.width / 50 - 1) * 50;
      let height = Math.round(props.height / 50 - 1) * 50;
      const yscale = d3
        .scaleLinear()
        .domain([0, height / scale.ratio])
        .range([height, 0]);
      const xscale = d3
        .scaleLinear()
        .domain([0, width / scale.ratio])
        .range([0, width]);
      const yAxis = d3.axisRight(yscale).ticks(1).tickSize(2);
      const xAxis = d3.axisTop(xscale).ticks(1).tickSize(2);
      s.append("g")
        .attr("transform", `translate(0, 25)`)
        .call(yAxis)
        .selectAll("text")
        .style("font-size", "7px");
      s.append("g")
        .attr("transform", `translate(25, ${props.height})`)
        .call(xAxis)
        .selectAll("text")
        .style("font-size", "7px");
      s.append("g")
        .append("path")
        .attr(
          "d",
          d3.line()([
            [0, 0],
            [0, props.height],
            [props.width, props.height],
            [props.width, 0],
            [0, 0],
          ])
        )
        .attr("stroke", "black")
        .attr("fill", "none");
      s.append("text")
        .attr(
          "transform",
          `translate(${props.width - 10}, ${props.height - 5})`
        )
        .style("text-anchor", "middle")
        .style("font-size", "10px")
        .text(scale.unit);
    };

    cache.current.createChart = () => {
      if (cache.current.chart) cache.current.chart.dispose();
      cache.current.chart = echarts.init(container.current);
      let points = [];
      echarts.registerMap("bg", { svg: svg.current });
      let option = {
        toolbox: {
          show: true,
          itemSize: 15,
          showTitle: false,
          feature: {
            saveAsImage: {
              show: true,
              excludeComponents: ["toolbox"],
              pixelRatio: 5,
            },
          },
        },
        geo: {
          map: "bg",
          center: [props.width / 2, props.height / 2],
          boundingCoords: [
            [-10, -10],
            [props.width + 10, props.height + 10],
          ],
          roam: true,
          zoom: 1.2,
        },
        visualMap: {
          show: false,
          top: "top",
          min: 0,
          max: 5,
          seriesIndex: 0,
          calculable: true,
          inRange: {
            color: ["blue", "blue", "green", "yellow", "red"],
          },
        },
        series: [
          {
            type: "heatmap",
            coordinateSystem: "geo",
            data: points,
            pointSize: 5,
            blurSize: 6,
          },
        ],
      };
      console.log(props.data, props.trackID)
      props.data.data[props.trackID].forEach(x => {
        const pt = x[0];
        if (pt[0] < 0) return
        points.push([...pt, 1])
      })
      cache.current.chart.setOption(option);
    };

    cache.current.createAxis(scale);
    cache.current.createChart();
  }, [props.data, props.height, props.trackID, props.width, scale]);

  React.useEffect(() => {
    if (container.current && cache.current.chart) {
      let observer = new ResizeObserver(() => {
        cache.current.chart.resize();
      })
      observer.observe(container.current)
    }
  }, [container.current, cache.current])

  React.useEffect(() => {
    if (scale.ratio !== props.scale.ratio || scale.unit !== props.scale.unit) {
      setScale(props.scale);
    }
  }, [props.scale, scale]);

  return (
    <div
      ref={container}
      style={{
        minWidth: 200,
        minHeight: props.height * (200 / props.width),
        aspectRatio: `${props.width}/${props.height}`,
      }}
    >
      <svg ref={svg}></svg>
    </div>
  );
};

const mapStateToProps = (state, props) => {
  return {
    width: props.videoMetaData.width,
    height: props.videoMetaData.height,
    scale: state.scale,
    ...props
  };
};

const mapDispatchToProps = null;

const VisibleHeatMap = connect(mapStateToProps, mapDispatchToProps)(HeatMap);

export default VisibleHeatMap;
