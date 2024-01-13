import React from "react";
import styles from "./ProgressBar.module.css";

const PROGRESS_BAR_BORDER_LENGTH = 100;

export default class ProgressBar extends React.Component {
  constructor(props) {
    super(props);
    this.state = { ratio: 1, settingTime: false };
    this.svgProgressBar = React.createRef();
    this.svgCursor = React.createRef();
    this.cursor = React.createRef();
    this.selectArea = React.createRef();
    // if (props.setCropTime) {
    // }
    this.isDrawing = false;
    this.handleMouseDown = this.handleMouseDown.bind(this);
    this.handleMouseMove = this.handleMouseMove.bind(this);
    this.handleMouseUp = this.handleMouseUp.bind(this);
  }

  componentDidMount() {
    if (this.props.cropTime) this.props.setCropTime([0, this.props.frames]);
    const updateRatio = () => {
      this.setState({
        ratio:
          this.svgCursor.current.width.baseVal.value /
          this.svgCursor.current.height.baseVal.value,
      });
    };
    new ResizeObserver(updateRatio).observe(this.svgCursor.current);
  }

  handleMouseMove(e) {
    this.pt.x = e.clientX;
    this.pt.y = e.clientY;
    let loc = this.pt.matrixTransform(
      this.svgCursor.current.getScreenCTM().inverse()
    );
    this.cursor.current.setAttribute("transform", `translate(${loc.x},0)`);
    this.lowerPt.x = e.clientX;
    this.lowerPt.y = e.clientY;
    let lowerLoc = this.lowerPt.matrixTransform(
      this.svgProgressBar.current.getScreenCTM().inverse()
    );
    this.props.setVideoTime((lowerLoc.x / 100) * this.props.duration);
    if (this.isDrawing) {
      this.props.pause();
      if (Math.abs(lowerLoc.x - this.lowerPt.x) > 1)
        this.drawRect(this.lowerPt.start.x, lowerLoc.x);
    }
  }

  handleMouseDown(e) {
    if (!this.isDrawing) {
      this.setState({ settingTime: true });
      this.pt = this.svgCursor.current.createSVGPoint();
      this.lowerPt = this.svgProgressBar.current.createSVGPoint();

      this.pt.x = e.clientX;
      this.pt.y = e.clientY;
      let loc = this.pt.matrixTransform(
        this.svgCursor.current.getScreenCTM().inverse()
      );
      this.cursor.current.setAttribute(
        "transform",
        `translate(${loc.x * this.state.ratio},0)`
      );
      if (this.props.setCropTime) this.isDrawing = true;
      this.lowerPt.x = e.clientX;
      this.lowerPt.y = e.clientY;
      let lowerLoc = this.lowerPt.matrixTransform(
        this.svgProgressBar.current.getScreenCTM().inverse()
      );
      this.lowerPt.start = lowerLoc;

      this.svgCursor.current.addEventListener(
        "mousemove",
        this.handleMouseMove
      );
      this.props.setVideoTime((lowerLoc.x / 100) * this.props.duration);
      this.svgCursor.current.addEventListener("mouseup", this.handleMouseUp);
    }
    // else if (!this.props.setCropTime) {
    //   this.pt = this.svgCursor.current.createSVGPoint();
    //   this.lowerPt = this.svgProgressBar.current.createSVGPoint();

    //   this.pt.x = e.clientX;
    //   this.pt.y = e.clientY;
    //   let loc = this.pt.matrixTransform(
    //     this.svgCursor.current.getScreenCTM().inverse()
    //   );
    //   this.cursor.current.setAttribute("transform", `translate(${loc.x},0)`);

    //   this.isDrawing = true;
    //   this.lowerPt.x = e.clientX;
    //   this.lowerPt.y = e.clientY;
    //   let lowerLoc = this.lowerPt.matrixTransform(
    //     this.svgProgressBar.current.getScreenCTM().inverse()
    //   );
    //   console.log(lowerLoc.x, this.props.duration);
    //   this.props.setVideoTime((lowerLoc.x / 100) * this.props.duration);
    // }
  }

  handleMouseUp(e) {
    this.isDrawing = false;
    this.lowerPt.x = e.clientX;
    this.lowerPt.y = e.clientY;
    let lowerLoc = this.pt.matrixTransform(
      this.svgProgressBar.current.getScreenCTM().inverse()
    );
    this.cursor.current.setAttribute(
      "transform",
      `translate(${lowerLoc.x * this.state.ratio},0)`
    );
    this.lowerPt.end = lowerLoc;
    this.svgCursor.current.removeEventListener("mouseup", this.handleMouseUp);
    this.props.setVideoTime((this.lowerPt.end.x / 100) * this.props.duration);

    if (this.props.setCropTime) {
      let start = this.lowerPt.start.x;
      let end = this.lowerPt.end.x;
      if (start > end) [start, end] = [end, start];
      if (end - start) {
        this.props.setCropTime([
          Math.round((start / 100) * this.props.frames),
          Math.round((end / 100) * this.props.frames),
        ]);
      }
    }

    this.svgCursor.current.removeEventListener(
      "mousemove",
      this.handleMouseMove
    );
    this.svgCursor.current.removeEventListener(
      "mousedown",
      this.handleMouseDown
    );
    this.setState({ settingTime: false });
  }

  // componentWillReceiveProps(nextProps) {
  //   let ratio =
  //     this.svgCursor.current.width.baseVal.value /
  //     this.svgCursor.current.height.baseVal.value;
  //   if (ratio !== this.state.ratio) this.setState({ ratio: ratio });
  // }

  drawRect(start, end) {
    let svg = this.selectArea.current;
    while (svg.lastChild) svg.removeChild(svg.lastChild);
    let rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    if (start > end) {
      [start, end] = [end, start];
    }
    let width = end - start;
    rect.setAttribute("x", start);
    rect.setAttribute("y", 0);
    rect.setAttribute("width", width);
    rect.setAttribute("height", PROGRESS_BAR_BORDER_LENGTH);
    rect.setAttribute("style", "fill: blue");
    svg.appendChild(rect);
    this.svgCursor.current.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      this.props.setCropTime([0, this.props.frames]);
      while (svg.lastChild) svg.removeChild(svg.lastChild);
    });
  }

  render() {
    const rectProps = {
      x: 0,
      y: 0,
      width: PROGRESS_BAR_BORDER_LENGTH,
      height: PROGRESS_BAR_BORDER_LENGTH,
    };
    return (
      <div className={styles.svgContainer}>
        <svg
          viewBox={`0 0 ${PROGRESS_BAR_BORDER_LENGTH} ${PROGRESS_BAR_BORDER_LENGTH}`}
          style={{ position: "absolute" }}
          className={styles.svgCanvas}
          preserveAspectRatio="xMinYMin"
          onMouseDown={this.handleMouseDown}
          ref={this.svgCursor}
        >
          <g
            ref={this.cursor}
            transform={
              this.state.settingTime
                ? ""
                : `translate(${
                    (this.props.currentTime / this.props.duration) *
                    100 *
                    this.state.ratio
                  }, 0)`
            }
          >
            <line x1={-10} y1={1} x2={10} y2={1} />
            <line x1={0} y1={0} x2={0} y2={PROGRESS_BAR_BORDER_LENGTH} />
            <line x1={-10} y1={99} x2={10} y2={99} />
          </g>
        </svg>
        <svg
          viewBox={`0 0 ${PROGRESS_BAR_BORDER_LENGTH} ${PROGRESS_BAR_BORDER_LENGTH}`}
          className={styles.svgCanvas}
          preserveAspectRatio="none"
          ref={this.svgProgressBar}
          style={{ pointerEvents: "all" }}
        >
          <g ref={this.selectArea}></g>
          <rect {...rectProps} style={{ fill: "#3F7A63", opacity: 0.5 }} />
        </svg>
      </div>
    );
  }
}
